/* Copyright 2021 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Range coder implementation, based on [1].
//
// [1] G. N. N. Martin, "Range coding: an algorithm for removing redundancy from
// a digitised message", presented to the Video & Data Recording Conference,
// held in Southampton, July 24-27, 1979.
//
#include "tensorflow_compression/cc/lib/range_coder.h"

#include <cstdint>
#include <limits>
#include <string>

#include "absl/base/integral_types.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow_compression {
void RangeEncoder::Encode(int32_t lower, int32_t upper, int precision,
                          std::string* sink) {
  // Input requirement: 0 <= lower < upper <= 2^precision.
  DCHECK_LE(0, lower);
  DCHECK_LT(lower, upper);
  DCHECK_LE(upper, 1 << precision);

  DCHECK_GT(precision, 0);
  DCHECK_LE(precision, 16);

  // `base` and `size` represent a half-open interval [base, base + size).
  // Loop invariant: 2^16 <= size <= 2^32.
  //
  // Note that keeping size above 2^16 is important. Since the interval sizes
  // are quantized to up to 16 bits, the smallest interval size the encode may
  // handle is 2^-16. If size is smaller than 2^16, a small interval input may
  // collapse the encoder range into an empty interval.
  const uint64_t size = static_cast<uint64_t>(size_minus1_) + 1;
  DCHECK_NE(size >> 16, 0);

  // For short notation, let u := lower and v := upper.
  //
  // The input u, v represents a half-open interval [u, v) / 2^precision.
  // This narrows the current interval roughly to
  // [base + (size * u) / 2^precision, base + (size * v) / 2^precision).
  //
  // TODO(ssjhv): Try rounding if it helps improve compression ratio, at the
  // expense of more operations. In the test using Zipf distribution, the
  // overhead over the theoretical compression ratio was ~0.01%.
  // NOTE: The max value of `size` is 2^32 and size > 0. Therefore `size * u`
  // can be rewritten as `(size - 1) * u + u` and all the computation can be
  // done in 32-bit mode. If 32-bit multiply is faster, then rewrite.
  const uint32_t a = (size * static_cast<uint64_t>(lower)) >> precision;
  const uint32_t b = ((size * static_cast<uint64_t>(upper)) >> precision) - 1;
  DCHECK_LE(a, b);

  // Let's confirm the RHS of a, b fit in uint32 type.
  // Recall that 0 <= u < 2^precision, and size <= 2^32. Therefore
  //   (size * u) / 2^precision < size <= 2^32,
  // and the value of a fits in uint32 type. Similarly, since v <= 2^precision,
  //   (size * v) / 2^precision - 1 <= size - 1 < 2^32.
  // For lower bound of b, note that 1 <= v, 2^16 <= size, and 16 <= precision.
  // Therefore (size * v) / 2^precision - 1 >= 2^16 / 2^precision - 1 >= 0.

  // The new interval is [base + a, base + b] = [base + a, base + b + 1).
  base_ += a;  // May overflow.
  size_minus1_ = b - a;
  const bool base_overflow = (base_ < a);

  // The encoder has two states. Let's call them state 0 and state 1.
  // State 0 is when base < base + size <= 2^32.
  // State 1 is when base < 2^32 < base + size.
  //
  // The encoder initially starts in state 0, with base = 0, size = 2^32.
  //
  // TODO(ssjhv): Requires some profiling, but the encoder stays in state 0
  // most of the time. Should optimize code for state 0.
  //
  // Each Encode() has up to two places where the interval changes:
  //   #1. Refine the interval. [base, base + size) -> [base + a, base + b + 1).
  //   #2. Expand interval if the new size is too small,
  // and each change may cause a state transition.
  //
  // First, consider when the current state is 0.
  //
  // In this case, the next state after #1 is always state 0, since refining
  // interval only shrinks the interval, therefore new_base + new_size <= 2^32.
  //
  // Let us explain #2.
  //
  // Recall that at the beginning of each Encode(), the encoder requires
  // 2^16 < size <= 2^32. As precision <= 16, the new interval size can be as
  // small as 1, but never zero.
  //
  // To keep size above 2^16, if new size is smaller than or equal to 2^16, the
  // encoder would left-shift base and size by 16 bits: size' <- size * 2^16.
  // Note that new size' is now in the range [2^16, 2^32].
  //
  // Since size is left-shifted, the same should be applied to base as well.
  // However, after the left-shift, base will then contain 48 bits instead of 32
  // bits. Therefore prior to the shift, The upper 16 bits in base should be
  // stored somewhere else.
  //
  // If the upper 16 bits of all values in the interval were the same, i.e., if
  // base[32:16] == (base + size - 1)[32:16], then base[32:16] can be written
  // out to `output` string, since any further Encode() only narrows down the
  // interval and that 16 bits would never change.
  //
  // If the upper 16 bits were not all the same, since this happens only when
  // size <= 2^16, the upper 16 bits may differ only by one, i.e.,
  // base[32:16] + 1 == (base + size - 1)[32:16]. At this stage, it is not
  // determined yet whether base[32:16] should be written to the output  or
  // (base[32:16] + 1) should be written to the output. In this case,
  // (base[32:16] + 1) is temporarily stored in `delay`, and base is
  // left-shifted by 16 bits.
  //
  // In the latter case, the condition implies that (base // 2^16) and
  // ((base + size - 1) // 2^16) were different. Therefore after left-shift by
  // 16 bits, the new (base + size) is greater than 2^32, i.e., the encoder
  // transition to state 1.
  //
  // ==== Summary ====
  // To detect the current encoder state,
  //   state 0: delay == 0 iff (base mod 2^32) < (base + size) mod 2^32,
  //   state 1: delay != 0 iff (base + size) mod 2^32 <= base mod 2^32,
  // because size <= 2^32.
  //
  // ==== Summary for state 0 ====
  // 1. Interval refinement does not cause state transition.
  // 2. Interval expansion may cause state transition, depending on the upper 16
  // bits of base and base + size - 1.
  //
  // Now suppose the previous state was 1. This means that
  // base <= 2^32 < base + size.
  //
  // When in state 1, an interval refinement may trigger state transition.
  // After Encode() refines the interval, there are three possibilities:
  //   #1. base <= 2^32 < base + size (unchanged),
  //   #2. 2^32 <= base < base + size (base overflowed),
  //   #3. base < base + size <= 2^32 (base + size - 1 underflowed).
  //
  // In case #1, the encoder remains in state 1.
  // In case #2 or #3, the encoder state changes to state 0.
  //
  // ==== State transition for interval refinement ====
  // 1. state 0 -> state 0,
  // 2. state 1 -> state 0 or state 1.
  //
  // Therefore if the new state is 1, then the previous state must have been
  // state 1.
  if (base_ + size_minus1_ < base_) {
    // If statement checked if 2^32 < base + size. The new state is 1, hence the
    // previous state was also state 1.
    DCHECK_NE(((base_ - a) + size) >> 32, 0);
    DCHECK_NE(delay_ & 0xFFFF, 0);

    // Like in state 0, if the new size is <= 2^16, then base and size should
    // be left-shifted by 16 bits. Combine the conditions
    // base <= 2^32 < base + size and size <= 2^16 to conclude that
    // base[32:16] >= 0xFFFF and (base + size - 1)[32:16] = 0x0000.
    //
    // Note that 2^32 - base < size, and since base is at least 0xFFFF0000,
    // 2^16 - base[16:0] < size. Let base' and size' be the new base and size
    // after the bit-shift. Then 2^32 - base' < size' => 2^32 < base' + size'.
    // Therefore the encoder remains in state 1.
    //
    // Lastly, `delay` is modified. Conceptually, delay has to be changed to
    //   delay' <- delay * 2^16 + (base + size - 1)[32:16].
    // Since we know above that (base + size - 1)[32:16] = 0x0000, there is no
    // need to explicitly do the computation above, but rather store how many
    // trailing zeros there were. For this reason, the lower 16 bits of
    // `delay` stores the delayed value when state changed from 0 to 1, and
    // delay[32:16] stores the # of trailing zeros (in bytes).
    //
    // ==== State transition for interval expansion ====
    // 1. state 0 -> state 0 or state 1,
    // 2. state 1 -> state 1.
    if (size_minus1_ >> 16 == 0) {
      DCHECK_EQ(base_ >> 16, 0xFFFF);
      base_ <<= 16;
      size_minus1_ <<= 16;
      size_minus1_ |= 0xFFFF;
      // TODO(ssjhv): It is possible that for very long input, delay
      // overflow during below. If overflow is detected, this delay is too
      // long the encoder should forcefully move to state 0. In such case,
      // base can be raised to 2^32 (force case #2), or (base + size) can be
      // lowered to 2^32 (force case #3), depending on which transition
      // keeps size larger.
      CHECK_LT(delay_, static_cast<uint64_t>(1) << 62);
      delay_ += 0x20000;  // Two more bytes of zeros. Check overflow?
    }
    return;
  }

  // If reached here, the current state is 0.
  // First handle the case when the previous state was state 1.
  if (delay_ != 0) {
    // In case #2 or #3, the encoder state changes to state 0. Recall that when
    // the encoder state changed from state 0 to state 1, the top 16 bits of
    // (base + size - 1) was temporarily stored in `delay`, because the output
    // could be either (delay - 1) or (delay).
    //
    // And from above, the delayed value encoded in `delay` is
    //   delay' <- delay[16:0] * 2^(8 * delay[MAX:16])
    //
    // In case #2, the interval moved below 2^32. So (delay' - 1) is the
    // converged value after interval refinements. Write out
    // (delay[16:0] - 1) and write (8 * delay[MAX:16]) bytes of 0xFF.
    //
    // In case #3, the interval moved above 2^32. So delay' is the converged
    // value after interval refinement. Write out delay[16:0] and write
    // (8 * delay[MAX:16]) bytes of 0x00.
    if (base_overflow) {
      // Case #2.
      DCHECK_NE((static_cast<uint64_t>(base_ - a) + a) >> 32, 0);
      sink->push_back(static_cast<char>(delay_ >> 8));
      sink->push_back(static_cast<char>(delay_ >> 0));
      sink->append(delay_ >> 16, static_cast<char>(0));
    } else {
      // Case #3.
      DCHECK_EQ(static_cast<uint64_t>(base_ + size_minus1_) >> 32, 0);
      --delay_;
      sink->push_back(static_cast<char>(delay_ >> 8));
      sink->push_back(static_cast<char>(delay_ >> 0));
      sink->append(delay_ >> 16, static_cast<char>(0xFF));
    }
    // Reset to state 0.
    delay_ = 0;
  }

  if (size_minus1_ >> 16 == 0) {
    const uint32_t top = base_ >> 16;

    base_ <<= 16;
    size_minus1_ <<= 16;
    size_minus1_ |= 0xFFFF;

    if (base_ <= base_ + size_minus1_) {
      // Still in state 0. Write the top 16 bits.
      sink->push_back(static_cast<char>(top >> 8));
      sink->push_back(static_cast<char>(top));
    } else {
      // New state is 1.
      DCHECK_LT(top, 0xFFFF);
      delay_ = top + 1;
    }
  }
}

void RangeEncoder::Finalize(std::string* sink) const {
  // Finalize the encode by writing out any number in the interval
  // [base, base + size).
  //
  // Trailing zeros are not explicitly written out as decoder can fill in zeros
  // by default.
  if (delay_ != 0) {
    // The last state was state 1. Since base < 2^32 < base + size, pick 2^32
    // (state 1, case #3).
    sink->push_back(static_cast<char>(delay_ >> 8));
    if ((delay_ & 0xFF) != 0) {
      sink->push_back(static_cast<char>(delay_));
    }
  } else if (base_ != 0) {
    // If base == 0, then pick 0 from [base, base + size) and no zeros are
    // explicitly written.

    // Otherwise, first check whether (base + (2^24 - base[24:0])) is inside the
    // interval, ie., rounding up base to the next multiple of 2^24 is within
    // [base, base + size). If true, then write the uppermost byte of the
    // round-up value.
    const uint32_t upper = base_ + size_minus1_;
    DCHECK_LE(base_, base_ + size_minus1_);  // Check state 0 consistency.
    // Rounding up to next multiple of 2^24 is (((base - 1) >> 24) + 1) << 24,
    // provided that base != 0.
    const uint32_t mid = ((base_ - 1) >> 24) + 1;
    if (mid <= (upper >> 24)) {
      sink->push_back(static_cast<char>(mid));
    } else {
      // Else, pick (base + (2^16 - base[16:0])), i.e., round up base to the
      // next multiple of 2^16. As 2^16 <= size, this value should be in the
      // interval [base, base + size).
      const uint32_t mid = ((base_ - 1) >> 16) + 1;
      DCHECK_EQ(mid & 0xFFFF, mid);
      sink->push_back(static_cast<char>(mid >> 8));
      // TODO(ssjhv): Is this ever false?
      if ((mid & 0xFF) != 0) {
        sink->push_back(static_cast<char>(mid >> 0));
      }
    }
  }
}

absl::Status RangeEncoder::CheckForError(int32_t lower, int32_t upper,
                                         int precision) const {
  if (ABSL_PREDICT_FALSE(!(0 < precision && precision <= 16))) {
    return absl::InvalidArgumentError(absl::StrCat(
        "precision not in (0, 16]: ", precision));
  }
  if (ABSL_PREDICT_FALSE(
          !(0 <= lower && lower < upper && upper <= (1 << precision)))) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Must satisfy 0 <= lower < upper <= ", 1 << precision,
        ": lower=", lower, ", upper=", upper
    ));
  }
  return absl::OkStatus();
}

template <typename T>
absl::Status RangeDecoder::CheckForErrorInternal(absl::Span<const T> cdf,
                                                 int precision,
                                                 bool allow_zero) const {
  if (ABSL_PREDICT_FALSE(!(0 < precision && precision <= 16))) {
    return absl::InvalidArgumentError(absl::StrCat(
        "precision not in (0, 16]: ", precision));
  }
  if (ABSL_PREDICT_FALSE(cdf.size() <= 1)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "cdf.size() = ", cdf.size(), " <= 1"));
  }

  for (typename absl::Span<const T>::size_type i = 0; i + 1 < cdf.size(); ++i) {
    bool monotonic = cdf[i] < cdf[i + 1];
    monotonic |= (allow_zero && cdf[i] == cdf[i + 1]);
    if (ABSL_PREDICT_FALSE(!monotonic)) {
      return absl::InvalidArgumentError("cdf is not monotonic");
    }
  }

  const T first = cdf[0];
  const T last = cdf[cdf.size() - 1];
  if (ABSL_PREDICT_FALSE(!(0 <= first && last <= (1 << precision)))) {
    return absl::InvalidArgumentError(absl::StrCat(
        "cdf values must be between 0 and ", 1 << precision));
  }

  const uint64_t size = static_cast<uint64_t>(size_minus1_) + 1;
  const uint64_t smallest = (static_cast<uint64_t>(value_ - base_) + 1)
                            << precision;
  if (ABSL_PREDICT_FALSE(!(size * static_cast<uint64_t>(first) < smallest))) {
    return absl::InvalidArgumentError(absl::StrCat(
        "cdf[0]=", first, " is too large and there was a decode error"));
  }
  if (ABSL_PREDICT_FALSE(!(smallest <= size * static_cast<uint64_t>(last)))) {
    return absl::InvalidArgumentError(absl::StrCat(
        "cdf[^1]=", last, " is too small and there was a decode error"));
  }

  return absl::OkStatus();
}

template absl::Status RangeDecoder::CheckForErrorInternal(
    absl::Span<const int16>, int, bool) const;
template absl::Status RangeDecoder::CheckForErrorInternal(
    absl::Span<const int32>, int, bool) const;
}  // namespace tensorflow_compression
