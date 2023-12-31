/* Copyright 2019-2021 Maxence Thevenet, Michael Rowan, Luca Fedeli, Axel Huebl
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef SHAPEFACTORS_H_
#define SHAPEFACTORS_H_

#include "Utils/TextMsg.H"

#include <AMReX.H>
#include <AMReX_GpuQualifiers.H>


/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specializations are defined for orders 0 to 3 (using "if constexpr").
 *  Shape factor functors may be evaluated with double arguments
 *  in current deposition to ensure that current deposited by
 *  particles that move only a small distance is still resolved.
 *  Without this safeguard, single and double precision versions
 *  can give disagreeing results in the time evolution for some
 *  problem setups.
 */
template <int depos_order>
struct Compute_shape_factor
{
    template< typename T >
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator()(
        T* const sx,
        T xmid) const
    {
        if constexpr (depos_order == 0){
            const auto j = static_cast<int>(xmid + T(0.5));
            sx[0] = T(1.0);
            return j;
        }
        else if constexpr (depos_order == 1){
            const auto j = static_cast<int>(xmid);
            const T xint = xmid - T(j);
            sx[0] = T(1.0) - xint;
            sx[1] = xint;
            return j;
        }
        else if constexpr (depos_order == 2){
            const auto j = static_cast<int>(xmid + T(0.5));
            const T xint = xmid - T(j);
            sx[0] = T(0.5)*(T(0.5) - xint)*(T(0.5) - xint);
            sx[1] = T(0.75) - xint*xint;
            sx[2] = T(0.5)*(T(0.5) + xint)*(T(0.5) + xint);
            // index of the leftmost cell where particle deposits
            return j-1;
        }
        else if constexpr (depos_order == 3){
            const auto j = static_cast<int>(xmid);
            const T xint = xmid - T(j);
            sx[0] = (T(1.0))/(T(6.0))*(T(1.0) - xint)*(T(1.0) - xint)*(T(1.0) - xint);
            sx[1] = (T(2.0))/(T(3.0)) - xint*xint*(T(1.0) - xint/(T(2.0)));
            sx[2] = (T(2.0))/(T(3.0)) - (T(1.0) - xint)*(T(1.0) - xint)*(T(1.0) - T(0.5)*(T(1.0) - xint));
            sx[3] = (T(1.0))/(T(6.0))*xint*xint*xint;
            // index of the leftmost cell where particle deposits
            return j-1;
        }
        else{
            WARPX_ABORT_WITH_MESSAGE("Unknown particle shape selected in Compute_shape_factor");
            amrex::ignore_unused(sx, xmid);
        }
        return 0;
    }
};



/**
 *  Compute shifted shape factor and return index of leftmost cell where
 *  particle writes, for Esirkepov algorithm.
 *  Specializations are defined below for orders 1, 2 and 3 (using "if constexpr").
 */
template <int depos_order>
struct Compute_shifted_shape_factor
{
    template< typename T >
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator()(
        T* const sx,
        const T x_old,
        const int i_new) const
    {
        if constexpr (depos_order == 0){
            const auto i = static_cast<int>(std::floor(x_old + T(0.5)));
            const int i_shift = i - i_new;
            sx[1+i_shift] = T(1.0);
            return i;
        }
        else if constexpr (depos_order == 1){
            const auto i = static_cast<int>(std::floor(x_old));
            const int i_shift = i - i_new;
            const T xint = x_old - T(i);
            sx[1+i_shift] = T(1.0) - xint;
            sx[2+i_shift] = xint;
            return i;
        }
        else if constexpr (depos_order == 2){
            const auto i = static_cast<int>(x_old + T(0.5));
            const int i_shift = i - (i_new + 1);
            const T xint = x_old - T(i);
            sx[1+i_shift] = T(0.5)*(T(0.5) - xint)*(T(0.5) - xint);
            sx[2+i_shift] = T(0.75) - xint*xint;
            sx[3+i_shift] = T(0.5)*(T(0.5) + xint)*(T(0.5) + xint);
            // index of the leftmost cell where particle deposits
            return i - 1;
        }
        else if constexpr (depos_order == 3){
            const auto i = static_cast<int>(x_old);
            const int i_shift = i - (i_new + 1);
            const T xint = x_old - i;
            sx[1+i_shift] = (T(1.0))/(T(6.0))*(T(1.0) - xint)*(T(1.0) - xint)*(T(1.0) - xint);
            sx[2+i_shift] = (T(2.0))/(T(3.0)) - xint*xint*(T(1.0) - xint/(T(2.0)));
            sx[3+i_shift] = (T(2.0))/(T(3.0)) - (T(1.0) - xint)*(T(1.0) - xint)*(T(1.0) - T(0.5)*(T(1.0) - xint));
            sx[4+i_shift] = (T(1.0))/(T(6.0))*xint*xint*xint;
            // index of the leftmost cell where particle deposits
            return i - 1;
        }
        else{
            WARPX_ABORT_WITH_MESSAGE("Unknown particle shape selected in Compute_shifted_shape_factor");
            amrex::ignore_unused(sx, x_old, i_new);
        }
        return 0;
    }
};

template<typename T>
struct shape_factor_precalc_result {
    T xint; // between -0.5 and 0.5
    int cell_start;
};

template<int depos_order, typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
shape_factor_precalc_result<T> shape_factor_precalc(const T xmid) {
    if constexpr (depos_order == 0) {
        const T i = std::floor(xmid + T(0.5));
        return {T(0.), static_cast<int>(i)};
    } else if constexpr (depos_order == 1) {
        const T i = std::floor(xmid);
        const T xint = xmid - (i + T(0.5));
        return {xint, static_cast<int>(i)};
    } else if constexpr (depos_order == 2) {
        const T i = std::floor(xmid + T(0.5));
        const T xint = xmid - i;
        return {xint, static_cast<int>(i) - 1};
    } else if constexpr (depos_order == 3) {
        const T i = std::floor(xmid);
        const T xint = xmid - (i + T(0.5));
        return {xint, static_cast<int>(i) - 1};
    }
    return {};
}

template<typename T>
struct shifted_shape_factor_precalc_result {
    T xint_old; // between -0.5 and 0.5
    T xint_new;
    int cell_start;
    int shift_old; // 0 or -1
    int shift_new;
    bool skip_end;
};

template<int depos_order, typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
shifted_shape_factor_precalc_result<T> shifted_shape_factor_precalc (const T xmid_old, const T xmid_new) {
    const auto so = shape_factor_precalc<depos_order>(xmid_old);
    const auto sn = shape_factor_precalc<depos_order>(xmid_new);
    const int cell_start = std::min(so.cell_start, sn.cell_start);
    const int shift_old = cell_start - so.cell_start;
    const int shift_new = cell_start - sn.cell_start;
    const bool skip_end = so.cell_start == sn.cell_start;
    return {so.xint, sn.xint, cell_start, shift_old, shift_new, skip_end};
}

template<int depos_order, typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
T branchless_shape_factor (const T xint, const int ic) {
    if constexpr (depos_order == 0) {
        return (0 <= ic && ic <= depos_order) ? T(1.0) : T(0.0);
    } else if constexpr (depos_order == 1) {
        const T xintm = (ic == 0) ? -xint : xint;
        const T ret = ((T(1.0)/T(2.0)) + xintm);
        return (0 <= ic && ic <= depos_order) ? ret : T(0.0);
    } else if constexpr (depos_order == 2) {
        const T xintm = (ic == 0) ? -xint : xint;
        const T xintm_2 = xintm * xintm;
        const T inner = (T(3.0)/T(4.0)) - xintm_2;
        const T outer = (T(1.0)/T(8.0)) + (T(1.0)/T(2.0))*(xintm + xintm_2);
        const T ret = ((ic == 1) ? inner : outer);
        return (0 <= ic && ic <= depos_order) ? ret : T(0.0);
    } else if constexpr (depos_order == 3) {
        const T xintm = ((ic & 1) == 0) ? -xint : xint;
        const T xintm_2 = xintm * xintm;
        const T xintm_3 = xintm_2 * xintm;
        const bool is_inner = (ic == 1 || ic == 2);
        const T ret = (
            (is_inner ? (T(23.0)/T(48.0)) : (T(1.0)/T(48.0)))
            + xintm * (is_inner ? (T(-5.0)/T(8.0)) : (T(1.0)/T(8.0)))
            + xintm_2 * (is_inner ? (T(-1.0)/T(4.0)) : (T(1.0)/T(4.0)))
            + xintm_3 * (is_inner ? (T(1.0)/T(2.0)) : (T(1.0)/T(6.0)))
        );
        return (0 <= ic && ic <= depos_order) ? ret : T(0.0);
    }
    return {};
}

template<typename T>
struct shifted_shape_factor_result {
    T s_old;
    T s_new;
};

template<int depos_order, typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
shifted_shape_factor_result<T> shifted_shape_factor (shifted_shape_factor_precalc_result<T> sp, int ic) {
    return {
        branchless_shape_factor<depos_order, T>(sp.xint_old, ic + sp.shift_old),
        branchless_shape_factor<depos_order, T>(sp.xint_new, ic + sp.shift_new)
    };
}

template<int depos_order, typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<T, depos_order+1> get_inner_shape_factor (shifted_shape_factor_precalc_result<T> sp) {
    amrex::GpuArray<T, depos_order+1> arr{};
    T sdx = T(0.0);
    for (int i=0; i<=depos_order; ++i) {
        auto [sx_old, sx_new] = shifted_shape_factor<depos_order>(sp, i);
        sdx += (sx_old - sx_new);
        arr[i] = sdx;
    }
}

#endif // SHAPEFACTORS_H_
