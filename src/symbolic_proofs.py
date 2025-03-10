
from grid import *

def main():
    sympy.init_printing(use_unicode=True)

    grid = Grid((2, 2, 2), (1, 1, 1))

    # cube.values = [0.1, -0.3, -0.6, 1.0, -0.5, 0.4, 0.7, -0.9]
    cube = grid.cell((0, 0, 0))
    cube.values = [sympy.symbols(f'f{i}') for i in range(8)]
    f0, f1, f2, f3, f4, f5, f6, f7 = cube.values
    # print(cube.asymptotes(0, iso=sympy.symbols('i0')))

    u, v, w = sympy.symbols('u,v,w')

    a, b, c, d, e, f, g, h = sympy.symbols('a,b,c,d,e,f,g,h')

    # compute saddle points
    # w = (a * u * v + b * u + c * v + d) / (e * u * v + f * u + g * v + h)
    a = f0 - f1 - f2 + f3
    b = -f0 + f1
    c = -f0 + f2
    d = f0
    e = f0 - f1 - f2 + f3 - f4 + f5 + f6 - f7
    f = -f0 + f1 + f4 - f5
    g = -f0 + f2 + f4 - f6
    h = f0 - f4
    # test = (a * h - b * g - c * f + d * e) * (-a * h + b * g + c * f - d * e)
    # tmp = sympy.expand(test)
    # det = a ** 2 * h ** 2 - 2 * a * b * g * h - 2 * a * c * f * h - 2 * a * d * e * h + 4 * a * d * f * g + b ** 2 * g ** 2 + 4 * b * c * e * h - 2 * b * c * f * g - 2 * b * d * e * g + c ** 2 * f ** 2 - 2 * c * d * e * f + d ** 2 * e ** 2
    # simpl = sympy.factor(det)
    # num_u = -a * h + b * g - c * f + d * e
    # num_v = -a * h - b * g + c * f + d * e
    # den_u = 2 * (a * f - b * e)
    # den_v = 2 * (a * g - c * e)
    #

    # solutions for saddle points
    # u1 = (-a * h + b * g - c * f + d * e - sympy.sqrt(
    #         a ** 2 * h ** 2 - 2 * a * b * g * h - 2 * a * c * f * h - 2 * a * d * e * h + 4 * a * d * f * g + b ** 2 * g ** 2 + 4 * b * c * e * h - 2 * b * c * f * g - 2 * b * d * e * g + c ** 2 * f ** 2 - 2 * c * d * e * f + d ** 2 * e ** 2)) / (2 * (a * f - b * e))
    # v1 = (-a * h - b * g + c * f + d * e - sympy.sqrt(
    #     a ** 2 * h ** 2 - 2 * a * b * g * h - 2 * a * c * f * h - 2 * a * d * e * h + 4 * a * d * f * g + b ** 2 * g ** 2 + 4 * b * c * e * h - 2 * b * c * f * g - 2 * b * d * e * g + c ** 2 * f ** 2 - 2 * c * d * e * f + d ** 2 * e ** 2)) / (2 * (a * g - c * e))
    # u2 = (-a * h + b * g - c * f + d * e + sympy.sqrt(
    #     a ** 2 * h ** 2 - 2 * a * b * g * h - 2 * a * c * f * h - 2 * a * d * e * h + 4 * a * d * f * g + b ** 2 * g ** 2 + 4 * b * c * e * h - 2 * b * c * f * g - 2 * b * d * e * g + c ** 2 * f ** 2 - 2 * c * d * e * f + d ** 2 * e ** 2)) / (2 * (a * f - b * e))
    # v2 = (-a * h - b * g + c * f + d * e + sympy.sqrt(
    #     a ** 2 * h ** 2 - 2 * a * b * g * h - 2 * a * c * f * h - 2 * a * d * e * h + 4 * a * d * f * g + b ** 2 * g ** 2 + 4 * b * c * e * h - 2 * b * c * f * g - 2 * b * d * e * g + c ** 2 * f ** 2 - 2 * c * d * e * f + d ** 2 * e ** 2)) / (2 * (a * g - c * e))
    # cost0 = sympy.count_ops([u1, v1, u2, v2], visual=True)
    # print(cost0)
    # u1 = sympy.simplify(u1)
    # v1 = sympy.simplify(v1)
    # u2 = sympy.simplify(u2)
    # v2 = sympy.simplify(v2)
    # cost1 = sympy.count_ops([u1, v1, u2, v2], visual=True)
    # print(cost1)
    # opt = sympy.cse([u1, v1, u2, v2])
    # cost2 = sympy.count_ops(opt, visual=True)
    # print(cost2)
    #
    # for subexp in opt[0]:
    #     print(sympy.python(subexp[1]))
    #
    # print(sympy.ccode(opt[-1][0]))
    # sys.exit(0)

    # verify saddles
    # inter = cube.trilinear((u, v, w))
    # w = sympy.solve(inter, w)[0]
    # w = sympy.collect(w, [u, v, u*v])
    # du = sympy.diff(w, u)
    # du = sympy.simplify(sympy.collect(sympy.simplify(sympy.expand(du)), [u, v, u * v]))
    # dv = sympy.diff(w, v)
    # dv = sympy.simplify(sympy.collect(sympy.simplify(sympy.expand(dv)), [u, v, u * v]))
    # res = sympy.solve([du, dv], [u,v], dict=True)
    # print(res)
    # sys.exit(0)

    # bilinear on face
    # inter = cube.trilinear((u, v, w))
    # inter = inter.subs(w, 0)
    # inter_v = sympy.solve(inter, v)[0]
    # inter_v = sympy.collect(inter_v, [u])
    # u_0 = sympy.solve(1 / inter_v, u)[0]
    # inter_u = sympy.solve(inter, u)[0]
    # inter_u = sympy.collect(inter_u, [v])
    # v_0 = sympy.solve(1 / inter_u, v)[0]
    # alpha = inter.subs(u, u_0)
    # alpha = alpha.subs(v, v_0)
    # alpha = sympy.simplify(alpha)
    # eta = sympy.symbols("eta")
    # normal_form = alpha + eta * (u - u_0) * (v - v_0)
    # eta = sympy.solve(normal_form - inter, eta)[0]

    # # degenerate to linear
    # inter_uv = sympy.collect(sympy.expand(inter), [u, v, u*v])
    # linear = inter_uv.subs(eta, 0)
    # sys.exit(0)

    # ({u: (-a * h + b * g - c * f + d * e - sqrt(
    #     a ** 2 * h ** 2 - 2 * a * b * g * h - 2 * a * c * f * h - 2 * a * d * e * h + 4 * a * d * f * g + b ** 2 * g ** 2 + 4 * b * c * e * h - 2 * b * c * f * g - 2 * b * d * e * g + c ** 2 * f ** 2 - 2 * c * d * e * f + d ** 2 * e ** 2)) / (2 * (a * f - b * e)),
    #  v: (-a * h - b * g + c * f + d * e - sqrt(
    #     a ** 2 * h ** 2 - 2 * a * b * g * h - 2 * a * c * f * h - 2 * a * d * e * h + 4 * a * d * f * g + b ** 2 * g ** 2 + 4 * b * c * e * h - 2 * b * c * f * g - 2 * b * d * e * g + c ** 2 * f ** 2 - 2 * c * d * e * f + d ** 2 * e ** 2)) / (2 * (a * g - c * e))},
    #  {u: (-a * h + b * g - c * f + d * e + sqrt(
    #     a ** 2 * h ** 2 - 2 * a * b * g * h - 2 * a * c * f * h - 2 * a * d * e * h + 4 * a * d * f * g + b ** 2 * g ** 2 + 4 * b * c * e * h - 2 * b * c * f * g - 2 * b * d * e * g + c ** 2 * f ** 2 - 2 * c * d * e * f + d ** 2 * e ** 2)) / (2 * (a * f - b * e)),
    #  v: (-a * h - b * g + c * f + d * e + sqrt(
    #     a ** 2 * h ** 2 - 2 * a * b * g * h - 2 * a * c * f * h - 2 * a * d * e * h + 4 * a * d * f * g + b ** 2 * g ** 2 + 4 * b * c * e * h - 2 * b * c * f * g - 2 * b * d * e * g + c ** 2 * f ** 2 - 2 * c * d * e * f + d ** 2 * e ** 2)) / (2 * (a * g - c * e))})

    # para = cube.parametric_level_set((u, v, w), 2, 0)
    # print(para)
    # grad = sympy.matrices.Matrix([[sympy.diff(para, u)], [sympy.diff(para, v)]])
    # du = sympy.diff(para, u)
    # dv = sympy.diff(para, v)
    # tmp0 = sympy.solve(du, u)
    # tmp1 = sympy.solve(dv, u)
    # extr = sympy.solve(tmp0 - tmp1)
    # extrema = grad.solve(sympy.matrices.Matrix([[0], [0]]))
    # print(extrema)

    # 3D extrema
    # u, v, w, i0 = sympy.symbols('u v w i0')
    # interpolation = cube.trilinear((u, v, w))
    # print(interpolation)
    #
    # diff = sympy.diff(interpolation, u, v, w)
    # print(diff)
    # extrema = sympy.solve(diff, u, v, w)
    # print(extrema)

    # compute line equation
    # x, a1, a2, a3, b1, b2, b3 = sympy.symbols('x a1 a2 a3 b1 b2 b3')
    # diagonal = interpolation.subs(u, a1 * (1 - x) + b1 * x, doit=False)
    # diagonal = diagonal.subs(v, a2 * (1 - x) + b2 * x, doit=False)
    # diagonal = diagonal.subs(w, a3 * (1 - x) + b3 * x, doit=False)
    # print(diagonal)
    # intersections = sympy.solve(diagonal, x)
    # print(intersections)
    # simpl = intersections[0].simplify()
    # print(simpl)
    # sys.exit(0)

    # asymptotes
    # parametric = cube.parametric_level_set((u, v, w), 0)
    # print(parametric)
    #
    # print(cube.bilinear_face((v, w), 5))
    # print(cube.hyperbola(u, 0))
    # print(cube.asymptotes(0))
    # print(cube.asymptotic_decider(0))
    # print(cube.hyperbola_intersections(2))

    # compare results with paper

    # quadratic equations
    # i0 = 0
    # a = (f5 - f4) * (f0 + f3 - f1 - f2) - (f1 - f0) * (f4 + f7 - f5 - f6)
    # b = (i0 - f0) * (f4 + f7 - f5 - f6) - (f1 - f0) * (f6 - f4) - (i0 - f4) * (f0 + f3 - f1 - f2) + (f5 - f4) * (
    #             f2 - f0)
    # c = (i0 - f0) * (f6 - f4) - (i0 - f4) * (f2 - f0)
    # det = b ** 2 - 4 * a * c
    # print(sympy.solve(det, i0))
    # quadratic_equations = a * u ** 2 + b * u + c
    # print(sympy.Eq(sympy.solve(quadratic_equations, u)[0], cube.hyperbola_intersections(2, i0)[0]))
    # print(sympy.Eq(sympy.solve(quadratic_equations, u)[1], cube.hyperbola_intersections(2, i0)[1]))


if __name__ == '__main__':
    main()
