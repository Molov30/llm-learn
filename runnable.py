import cmath
import math

from langchain_core.runnables import RunnableBranch, RunnableLambda


def calc_discriminant(coef: dict[str, float]) -> dict[str, float]:
    d = coef["b"] ** 2 - 4 * coef["a"] * coef["c"]
    coef["D"] = d
    return coef


def calc_two_root(coef: dict[str, float]) -> dict[str, float]:
    x1 = (-coef["b"] + math.sqrt(coef["D"])) / (2 * coef["a"])
    x2 = (-coef["b"] - math.sqrt(coef["D"])) / (2 * coef["a"])
    return {"x1": x1, "x2": x2}


def calc_one_root(coef: dict[str, float]) -> dict[str, float]:
    return {"x": -coef["b"] / 2 * coef["a"]}


def calc_complex_roots(coef: dict[str, float]) -> dict[str, complex]:
    a, b, D = coef["a"], coef["b"], coef["D"]
    sqrt_d = cmath.sqrt(D)
    x1 = (-b + sqrt_d) / (2 * a)
    x2 = (-b - sqrt_d) / (2 * a)
    return {"x1": x1, "x2": x2}


branch = RunnableBranch(
    (
        lambda x: x["D"] > 0,  # pyright: ignore[reportIndexIssue]
        RunnableLambda(calc_two_root),
    ),
    (
        lambda x: x["D"] == 0,  # pyright: ignore[reportIndexIssue]
        RunnableLambda(calc_one_root),
    ),
    RunnableLambda(calc_complex_roots),
)

pipeline = RunnableLambda(calc_discriminant) | branch

print(pipeline.invoke({"a": 1, "b": 6, "c": -4}))
print(pipeline.get_graph().print_ascii())
