# https://github.com/volkerp/fitCurves

from typing import List

import bezier
import numpy as np
import numpy.typing as npt


def hodo(p: npt.NDArray, degree: int = 3) -> npt.NDArray:
    return degree * (p[1:] - p[:-1])


def q(p: npt.NDArray, t: npt.NDArray) -> npt.NDArray:
    """evaluates bezier at t"""
    return bezier.Curve.from_nodes(p.T).evaluate_multi(t).T


def qprime(p: npt.NDArray, t: npt.NDArray) -> npt.NDArray:
    """evaluates bezier first derivative at t"""
    return bezier.Curve.from_nodes(hodo(p).T).evaluate_multi(t).T


def qprimeprime(p: npt.NDArray, t: npt.NDArray) -> npt.NDArray:
    """evaluates bezier second derivative at t"""
    return bezier.Curve.from_nodes(hodo(hodo(p)).T).evaluate_multi(t).T


def normalize(v: npt.NDArray) -> npt.NDArray:
    magnitude = np.sqrt(np.dot(v, v))
    if magnitude < np.finfo(float).eps:
        return v
    return v / magnitude


def compute_error(
    p: npt.NDArray,
    points: npt.NDArray,
    u: npt.NDArray,
) -> npt.NDArray:
    errs = ((q(p, u) - points) ** 2).sum(-1)
    split_point = errs.argmax()

    # Prevent split_point from being at the ends
    if split_point == 0 or split_point == len(points) - 1:
        split_point = len(points) // 2  # Split in the middle

    return float(errs[split_point]), int(split_point)


def get_segment_length(p: npt.NDArray) -> float:
    return float(bezier.Curve.from_nodes(p.T).length)


def fit_bezier(
    points: npt.NDArray,
    max_err: float,
    left_tangent: npt.NDArray = None,
    right_tangent: npt.NDArray = None,
) -> List[npt.NDArray]:
    """fit one (or more) Bezier curves to a set of points"""

    if len(points) < 2:
        return []

    weights = (lambda x, n: (x ** -np.arange(1, n + 1)) / (1 - x**-n) * (x - 1))(2.0, min(5, len(points) - 2))

    if left_tangent is None:
        # points[1] - points[0]
        l_vecs = points[2 : 2 + len(weights)] - points[1]
        left_tangent = normalize(np.einsum("np,n->p", l_vecs, weights))

    if right_tangent is None:
        # points[-2] - points[-1]
        r_vecs = points[-3 : -3 - len(weights) : -1] - points[-2]
        right_tangent = normalize(np.einsum("np,n->p", r_vecs, weights))

    if points.shape[0] == 2:
        return [points]

    # parameterize points, assuming constant speed
    u = np.cumsum(np.linalg.norm(points[1:] - points[:-1], axis=1))
    u = np.pad(u, (1, 0)) / u[-1]

    for _ in range(32):
        bez_curve = generate_bezier(points, u, left_tangent, right_tangent)
        err, split_point = compute_error(bez_curve, points, u)

        if err < max_err:
            # check if line is a good fit
            line_err, _ = compute_error(bez_curve[[0, -1]], points, u)
            if line_err < max_err:
                return [bez_curve[[0, -1]]]

            return [bez_curve]

        u = newton_raphson_root_find(bez_curve, points, u)

    # Fitting failed -- split at max error point and fit recursively
    center_tangent = normalize(points[split_point - 1] - points[split_point + 1])
    return [
        *fit_bezier(points[: split_point + 1], max_err, left_tangent, center_tangent),
        *fit_bezier(points[split_point:], max_err, -center_tangent, right_tangent),
    ]


def generate_bezier(
    points: npt.NDArray,
    u: npt.NDArray,
    left_tangent: npt.NDArray,
    right_tangent: npt.NDArray,
) -> npt.NDArray:
    bez_curve = np.array([points[0], points[0], points[-1], points[-1]])

    # compute the A's
    S = 3 * (1 - u) * u
    A_l = (S * (1 - u))[:, np.newaxis] * left_tangent
    A_r = (S * u)[:, np.newaxis] * right_tangent
    A = np.stack([A_l, A_r], axis=1)

    # Create the C and X matrices
    C = np.einsum("nij,nkj->ik", A, A)
    d = points - q(bez_curve, u)
    X = np.einsum("nij,ni->j", A, d)

    try:
        alpha_l, alpha_r = np.linalg.lstsq(C, X, rcond=None)[0]
    except np.linalg.LinAlgError:
        alpha_l, alpha_r = 0.0, 0.0

    seg_len = np.linalg.norm(points[0] - points[-1])
    epsilon = 1e-6 * seg_len
    if alpha_l < epsilon or alpha_r < epsilon:
        bez_curve[1] = points[0] + left_tangent * (seg_len / 3.0)
        bez_curve[2] = points[-1] + right_tangent * (seg_len / 3.0)
    else:
        bez_curve[1] = points[0] + left_tangent * alpha_l
        bez_curve[2] = points[-1] + right_tangent * alpha_r

    return bez_curve


def newton_raphson_root_find(
    bez: npt.NDArray,
    points: npt.NDArray,
    u: npt.NDArray,
) -> npt.NDArray:
    """
    Newton's root finding algorithm calculates f(x)=0 by reiterating
    x_n+1 = x_n - f(x_n)/f'(x_n)
    We are trying to find curve parameter u for some point p that minimizes
    the distance from that point to the curve. Distance point to curve is d=q(u)-p.
    At minimum distance the point is perpendicular to the curve.
    We are solving
    f = q(u)-p * q'(u) = 0
    with
    f' = q'(u) * q'(u) + q(u)-p * q''(u)
    gives
    u_n+1 = u_n - |q(u_n)-p * q'(u_n)| / |q'(u_n)**2 + q(u_n)-p * q''(u_n)|
    """

    d = q(bez, u) - points
    qp = qprime(bez, u)
    num = (d * qp).sum(-1)
    den = (qp**2 + d * qprimeprime(bez, u)).sum(-1)

    return u - np.divide(num, den, out=np.zeros_like(num), where=den != 0)
