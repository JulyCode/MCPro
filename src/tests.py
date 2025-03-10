import random
import unittest
from mc import *
from test_utils import betti_0, betti_1, betti_2, isolated_vertices, duplicate_vertices, edge_manifold, vertex_manifold, \
    self_intersecting, orientable, count_boundary_components, euler_characteristics, duplicate_faces


class TopologyTest:
    class _TopologyTest(unittest.TestCase):
        def __init__(self, methodName='runTest'):
            super().__init__(methodName)
            np.set_printoptions(precision=18)
            self.values = [1] * 8
            self.solutions = []
            self.ref_betti = None
            self.is_edge_manifold = True
            self.is_vertex_manifold = True
            self.is_orientable = True

        def setUp(self):
            grid = Grid((2, 2, 2), (1, 1, 1))
            grid.values = self.values
            self.tmc = tmc(grid)
            self.ref = grid.cell((0, 0, 0)).trilinear_isosurface_mc(subdivision=101)

        def test_betti_numbers(self):
            if self.ref_betti is None or len(self.ref_betti) != 3:
                self.assertEqual(betti_0(self.tmc[0], self.tmc[1]), betti_0(self.ref[0], self.ref[1]))
                self.assertEqual(betti_1(self.tmc[0], self.tmc[1]), betti_1(self.ref[0], self.ref[1]))
                self.assertEqual(betti_2(self.tmc[0], self.tmc[1]), betti_2(self.ref[0], self.ref[1]))
            else:
                self.assertEqual(betti_0(self.tmc[0], self.tmc[1]), self.ref_betti[0])
                self.assertEqual(betti_1(self.tmc[0], self.tmc[1]), self.ref_betti[1])
                self.assertEqual(betti_2(self.tmc[0], self.tmc[1]), self.ref_betti[2])

        def test_properties(self):
            self.assertEqual(isolated_vertices(self.tmc[0], self.tmc[1]), isolated_vertices(self.ref[0], self.ref[1]))
            self.assertEqual(duplicate_vertices(self.tmc[0], self.tmc[1]), 0)
            self.assertEqual(duplicate_faces(self.tmc[0], self.tmc[1]), 0)
            self.assertFalse(self_intersecting(self.tmc[0], self.tmc[1]))

            self.assertEqual(edge_manifold(self.tmc[0], self.tmc[1]), self.is_edge_manifold)
            self.assertEqual(vertex_manifold(self.tmc[0], self.tmc[1]), self.is_vertex_manifold)
            self.assertEqual(orientable(self.tmc[0], self.tmc[1]), self.is_orientable)

            self.assertEqual(count_boundary_components(self.tmc[0], self.tmc[1]), count_boundary_components(self.ref[0], self.ref[1]))
            self.assertEqual(euler_characteristics(self.tmc[0], self.tmc[1]), euler_characteristics(self.ref[0], self.ref[1]))

        # def test_solutions(self):
        #     print(self.tmc)
        #     for sol_vertices, sol_faces in self.solutions:
        #         if (sol_vertices == self.tmc[0]).all() and (sol_faces == self.tmc[1]).all():
        #             return
        #     self.assertTrue(False)


def example_he_fail():
    # return [-0.48450005054473877, 0.68111252784729, -0.1387561559677124, -0.3911685049533844, 0.9471075534820557, 0.3874952495098114, 0.373460054397583, 0.5864385366439819]
    return [993.0, 9.0, 0.0, 31.0, 796.0, -97.0, -62.0, -31.0]

def example_head_fail_0():
    # return [0.0, 609.0, 0.0, 1712.0, 0.0, 468.0, 34.0, 1404.0]
    # cell (78, 116, 138)
    return [25.0, -4.0, -925.0, -950.0, 0.0, 36.0, -950.0, -950.0]

def example_no_decider_0():
    return [-1, 1, -1, 1, -1, 1, -1, 1]

class TestNoDecider0(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_no_decider_0()
        self.solutions = [( np.array([[0.5 , 0.  , 0.  ],
                                       [0.5 , 0.  , 1.  ],
                                       [0.5 , 1.  , 0.  ],
                                       [0.5 , 1.  , 1.  ]]),
                            np.array([[0, 2, 3],
                                       [0, 3, 1],
                                       [0, 1, 0]]))]

def example_no_decider_1():
    return [-1, -1, -1, -1, 1, -1, 1, -1]

class TestNoDecider1(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_no_decider_1()
        self.solutions = [(np.array([[0.  , 0.  , 0.5 ],
       [0.5 , 0.  , 1.  ],
       [0.  , 1.  , 0.5 ],
       [0.5 , 1.  , 1.  ]]), np.array([[1, 3, 2],
       [1, 2, 0],
       [1, 0, 1]]))]

def example_w_saddle_fail_0():
    return [2.884, 2.88578, 2.88664, 2.88913, 2.88388, 2.88596, 2.88666, 2.8891]

class WSaddleFail0(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_w_saddle_fail_0()
        self.is_orientable = False
        self.solutions = [(np.array([]), np.array([]))]

def example_w_saddle_fail_1():
    return [19.0, -5.0, 4.0, -22.0, 25.0, -5.0, 20.0, -22.0]

class WSaddleFail1(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_w_saddle_fail_1()
        self.solutions = [(np.array([[0.7916666666666666 , 0.                 , 0.                 ],
       [0.8333333333333334 , 0.                 , 1.                 ],
       [0.15384615384615385, 1.                 , 0.                 ],
       [0.47619047619047616, 1.                 , 1.                 ]]), np.array([[2, 0, 1],
       [2, 1, 3],
       [2, 3, 2]]))]

def example_w_saddle_fail_2():
    return [12.0, -9.999999999999993, -5.0, -28.999999999999993, 12.0, -5.999999999999986, -5.0, -34.999999999999986]

class WSaddleFail2(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_w_saddle_fail_2()
        self.solutions = [(np.array([[0.5454545454545456, 0.                , 0.                ],
       [0.                , 0.7058823529411765, 0.                ],
       [0.6666666666666672, 0.                , 1.                ],
       [0.                , 0.7058823529411765, 1.                ]]), np.array([[1, 0, 2],
       [1, 2, 3],
       [1, 3, 1]]))]

def example_outer_next_error():
    return [0.0, -12.0, 7.0, -5.0, 8.0, -19.0, 5.0, -26.0]

class OuterNextError(TopologyTest._TopologyTest):  # this is fine, fix test
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_outer_next_error()
        self.solutions = [(np.array([[0.0000000000000000e+00, 8.3540186821452864e-04,
        0.0000000000000000e+00],
       [0.0000000000000000e+00, 0.0000000000000000e+00,
        7.3105297506535414e-04],
       [2.9629629629629628e-01, 0.0000000000000000e+00,
        1.0000000000000000e+00],
       [5.8333333333333337e-01, 1.0000000000000000e+00,
        0.0000000000000000e+00],
       [1.6129032258064516e-01, 1.0000000000000000e+00,
        1.0000000000000000e+00],
       [2.3687254321044754e-01, 4.0644593477553165e-01,
        6.3956301673110472e-01]]), np.array([[3, 0, 1],
       [3, 1, 2],
       [3, 2, 5],
       [3, 5, 3],
       [2, 4, 5],
       [2, 5, 2],
       [4, 3, 5],
       [4, 5, 4]])),
        (np.array([[2.1882820703312955e-05, 0.0000000000000000e+00,
        0.0000000000000000e+00],
       [2.9629629629629628e-01, 0.0000000000000000e+00,
        1.0000000000000000e+00],
       [5.8333333333333337e-01, 1.0000000000000000e+00,
        0.0000000000000000e+00],
       [1.6129032258064516e-01, 1.0000000000000000e+00,
        1.0000000000000000e+00],
       [2.3691668060668428e-01, 4.0612588051353365e-01,
        6.3939905354499893e-01]]), np.array([[2, 0, 1],
       [2, 1, 4],
       [2, 4, 2],
       [1, 3, 4],
       [1, 4, 1],
       [3, 2, 4],
       [3, 4, 3]]))]

def example_outer_next_error_2():
    return [-52, 4, 46, -28, 5, 52, 46, 20]

class OuterNextError2(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_outer_next_error_2()
        self.solutions = [(np.array([[0.9285714285714286 , 0.                 , 0.                 ],
       [0.                 , 0.5306122448979592 , 0.                 ],
       [0.                 , 0.                 , 0.9122807017543859 ],
       [0.6216216216216216 , 1.                 , 0.                 ],
       [1.                 , 0.125              , 0.                 ],
       [1.                 , 1.                 , 0.5833333333333334 ],
       [0.7270871449350248 , 0.4604330005161073 , 0.22362200034407154]]), np.array([[0, 4, 6],
       [0, 6, 0],
       [3, 1, 6],
       [3, 6, 5],
       [3, 5, 3],
       [2, 0, 6],
       [2, 6, 1],
       [2, 1, 2],
       [4, 5, 6],
       [4, 6, 4]]))]

def example_outer_halfedge_none():  # this is wierd, looks singular but isnt
    return [-97, -49, -5.2281466127914096e-05, -6.164388934690916e-05, -62, -26, -56, 18]  # saddle point is outside, but rounded to inside. only 3 edge intersections, not singular. saddle points cant find outer halfedges

class OuterHalfedgeNone(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_outer_halfedge_none()
        self.solutions = []

def example_cross():
    return [1,-1,-1,1,-1,1,1,-1]

def example_cross_translated():
    return [0.2, -0.8, -0.2, 0.8, -0.2, 0.8, 0.2, -0.8]

def example_plane_with_hyperbolas():
    return [0.2, -0.8, -0.2, 0.8, -0.8, 0.2, 0.8, -0.2]

class Cross(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_cross()
        self.solutions = []

def example_complete_tunnel_cell():
    return [1, 0, 0, 0, 0, 0, 0, 1]

class CompleteTunnelCell(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_complete_tunnel_cell()
        self.solutions = []

def example_complete_hexagon_disc():
    return [1, 0, 0, 0, 0, 0, 0, -1]

class CompleteHexagonDisc(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_complete_hexagon_disc()
        self.solutions = []
        self.ref_betti = [1, 0, 1]

def example_MC_4_with_tunnel():
    return [x - -1.7660 for x in [-7.70146936482581, -3.21868369245987, -5.44023748418735, 15.6051950593180, 12.7611835388515, -4.46952393442309, -11.7240576326183, -9.23038948829007]]

def example_MC_4_without_tunnel():
    return [x for x in [-6, -1.5, -4, 17, 14, -3, -10, 8]]

class MC4WithTunnel(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_MC_4_with_tunnel()
        self.solutions = [(np.array([[0.                 , 0.                 , 0.2900635314866313 ],
       [0.8430984950202222 , 0.                 , 1.                 ],
       [0.                 , 0.5933036737158454 , 1.                 ],
       [0.1745859809054495 , 1.                 , 0.                 ],
       [1.                 , 0.07717238894362659, 0.                 ],
       [1.                 , 1.                 , 0.6994478034538966 ],
       [0.7391797867508412 , 0.18076970379206214, 0.5941978641402078 ],
       [0.7391797867508412 , 0.18076970379206214, 0.35364812684550967],
       [0.4070864317756304 , 0.18076970379206214, 0.35364812684550967],
       [0.4070864317756304 , 0.45654631818764824, 0.35364812684550967],
       [0.4070864317756304 , 0.45654631818764824, 0.5941978641402078 ],
       [0.7391797867508412 , 0.45654631818764824, 0.5941978641402078 ]]), np.array([[ 4,  3,  9],
       [ 4,  9,  8],
       [ 4,  8,  7],
       [ 4,  7,  4],
       [ 1,  2, 10],
       [ 1, 10, 11],
       [ 1, 11,  6],
       [ 1,  6,  1],
       [ 0,  1,  6],
       [ 0,  6,  7],
       [ 0,  7,  8],
       [ 0,  8,  0],
       [ 3,  5, 11],
       [ 3, 11, 10],
       [ 3, 10,  9],
       [ 3,  9,  3],
       [ 2,  0,  8],
       [ 2,  8,  9],
       [ 2,  9, 10],
       [ 2, 10,  2],
       [ 5,  4,  7],
       [ 5,  7,  6],
       [ 5,  6, 11],
       [ 5, 11,  5]]))]

def example_MC_7_with_tunnel():
    return [x - -0.6 for x in [-3.42744283804455, 0.621278122151001, 4.48110777981235, -1.95551129669134,2.30448107596369,-1.04182240925489,-3.51087814405650,-6.44976786808517]]

class MC7WithTunnel(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_MC_7_with_tunnel()
        self.solutions = [(np.array([[0.6983545830503434 , 0.                 , 0.                 ],
       [0.                 , 0.35751719558580064, 0.                 ],
       [0.                 , 0.                 , 0.49327989702280706],
       [0.8679670235510542 , 0.                 , 1.                 ],
       [0.                 , 0.4994499851298277 , 1.                 ],
       [0.789406320215606  , 1.                 , 0.                 ],
       [0.                 , 1.                 , 0.6357753665002241 ],
       [1.                 , 0.4739534062118578 , 0.                 ],
       [1.                 , 0.                 , 0.7343381227342893 ],
       [0.28963498341888505, 0.3396792936640344 , 0.46093624722039095],
       [0.28963498341888505, 0.3396792936640344 , 0.13181880852211156],
       [0.6740152775280839 , 0.3396792936640344 , 0.13181880852211156],
       [0.6740152775280839 , 0.11713589075971596, 0.13181880852211156],
       [0.6740152775280839 , 0.11713589075971596, 0.46093624722039095],
       [0.28963498341888505, 0.11713589075971596, 0.46093624722039095]]), np.array([[ 0,  1, 10],
       [ 0, 10, 11],
       [ 0, 11, 12],
       [ 0, 12,  0],
       [ 5,  7, 11],
       [ 5, 11,  5],
       [ 3,  4,  9],
       [ 3,  9, 14],
       [ 3, 14, 13],
       [ 3, 13,  3],
       [ 2,  0, 12],
       [ 2, 12, 13],
       [ 2, 13, 14],
       [ 2, 14,  2],
       [ 8,  3, 13],
       [ 8, 13,  8],
       [ 6,  5, 11],
       [ 6, 11, 10],
       [ 6, 10,  9],
       [ 6,  9,  6],
       [ 1,  2, 14],
       [ 1, 14,  9],
       [ 1,  9, 10],
       [ 1, 10,  1],
       [ 4,  6,  9],
       [ 4,  9,  4],
       [ 7,  8, 13],
       [ 7, 13, 12],
       [ 7, 12, 11],
       [ 7, 11,  7]]))]

def example_MC_10_with_tunnel():
    return [x - 1.10918 for x in [-0.100000000000000,-6.11000000000000,2,10.2000000000000,10.8000000000000,1.80000000000000,-8.20000000000000,-0.180000000000000]]

class MC10WithTunnel(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_MC_10_with_tunnel()
        self.solutions = [(np.array([[0.                 , 0.5758000000000001 , 0.                 ],
       [0.                 , 0.                 , 0.11093394495412845],
       [0.                 , 0.5100431578947369 , 1.                 ],
       [0.                 , 1.                 , 0.08733529411764705],
       [1.                 , 0.44262293071735137, 0.                 ],
       [1.                 , 0.                 , 0.9126649810366625 ],
       [1.                 , 0.3488989898989899 , 1.                 ],
       [1.                 , 1.                 , 0.8758015414258189 ],
       [0.8658956891382299 , 0.4452266669409782 , 0.7716562126556008 ],
       [0.8658956891382299 , 0.4452266669409782 , 0.18210855012936392],
       [0.11836183631118724, 0.4452266669409782 , 0.18210855012936392],
       [0.11836183631118724, 0.5078197783117268 , 0.18210855012936392],
       [0.11836183631118724, 0.5078197783117268 , 0.7716562126556008 ],
       [0.8658956891382299 , 0.5078197783117268 , 0.7716562126556008 ]]), np.array([[ 4,  0, 11],
       [ 4, 11, 10],
       [ 4, 10,  9],
       [ 4,  9,  4],
       [ 6,  2, 12],
       [ 6, 12, 13],
       [ 6, 13,  8],
       [ 6,  8,  6],
       [ 1,  5,  8],
       [ 1,  8,  9],
       [ 1,  9, 10],
       [ 1, 10,  1],
       [ 3,  7, 13],
       [ 3, 13, 12],
       [ 3, 12, 11],
       [ 3, 11,  3],
       [ 0,  3, 11],
       [ 0, 11,  0],
       [ 2,  1, 10],
       [ 2, 10, 11],
       [ 2, 11, 12],
       [ 2, 12,  2],
       [ 7,  4,  9],
       [ 7,  9,  8],
       [ 7,  8, 13],
       [ 7, 13,  7],
       [ 5,  6,  8],
       [ 5,  8,  5]]))]

def example_MC_12_with_tunnel():
    return [x - 0.0708 for x in [-3.37811990337124,0.473258332744286,2.54344310345736,7.87658724379480,4.38700713005133,-1.49950251870885,-4.21025867362045,-1.00233824192217]]

class MC12WithTunnel(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_MC_12_with_tunnel()
        self.solutions = [(np.array([[0.8955027763904584 , 0.                 , 0.                 ],
       [0.                 , 0.5824340464492282 , 0.                 ],
       [0.                 , 0.                 , 0.4441549878741763 ],
       [0.7332370772483847 , 0.                 , 1.                 ],
       [0.                 , 0.5020441648096927 , 1.                 ],
       [0.                 , 1.                 , 0.3661167142217557 ],
       [1.                 , 0.                 , 0.20400766390301953],
       [1.                 , 1.                 , 0.879136473929366  ],
       [0.6870893013354001 , 0.11566534066648096, 0.747144992623024  ],
       [0.6870893013354001 , 0.11566534066648096, 0.4541719255899158 ],
       [0.1361512247743526 , 0.11566534066648096, 0.4541719255899158 ],
       [0.1361512247743526 , 0.47760955318473897, 0.4541719255899158 ],
       [0.1361512247743526 , 0.47760955318473897, 0.747144992623024  ],
       [0.6870893013354001 , 0.47760955318473897, 0.747144992623024  ]]), np.array([[ 0,  1, 11],
       [ 0, 11, 10],
       [ 0, 10,  9],
       [ 0,  9,  0],
       [ 3,  4, 12],
       [ 3, 12, 13],
       [ 3, 13,  8],
       [ 3,  8,  3],
       [ 6,  0,  9],
       [ 6,  9,  6],
       [ 2,  3,  8],
       [ 2,  8,  9],
       [ 2,  9, 10],
       [ 2, 10,  2],
       [ 5,  7, 13],
       [ 5, 13, 12],
       [ 5, 12, 11],
       [ 5, 11,  5],
       [ 1,  5, 11],
       [ 1, 11,  1],
       [ 4,  2, 10],
       [ 4, 10, 11],
       [ 4, 11, 12],
       [ 4, 12,  4],
       [ 7,  6,  9],
       [ 7,  9,  8],
       [ 7,  8, 13],
       [ 7, 13,  7]]))]

def example_MC_13_with_tunnel():
    return [x - -1.3064 for x in [2.74742516087490, -3.39187542578189, -12.5297639669456, 0.431517989649243, -6.92460546400188, 2.52228314017858, 14.6950568276448, -10.0732624062474]]

class MC13WithTunnel(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_MC_13_with_tunnel()
        self.solutions = [(np.array([[0.6603073271384561 , 0.                 , 0.                 ],
       [0.                 , 0.2653515072018509 , 0.                 ],
       [0.                 , 0.                 , 0.41912865230682056],
       [0.5947149055526809 , 0.                 , 1.                 ],
       [0.                 , 0.2598655514694427 , 1.                 ],
       [0.8659146529279096 , 1.                 , 0.                 ],
       [0.                 , 1.                 , 0.4122474873801812 ],
       [0.6460453241311952 , 1.                 , 1.                 ],
       [1.                 , 0.5454514352001958 , 0.                 ],
       [1.                 , 0.                 , 0.3526241987803729 ],
       [1.                 , 0.30397120363436575, 1.                 ],
       [1.                 , 1.                 , 0.16544067787729339],
       [0.5590932972488663 , 0.13513676205463904, 0.6486955649765204 ],
       [0.5590932972488663 , 0.13513676205463904, 0.4245312994035552 ],
       [0.10942802149523169, 0.13513676205463904, 0.4245312994035552 ],
       [0.10942802149523169, 0.2564647590274203 , 0.4245312994035552 ],
       [0.10942802149523169, 0.2564647590274203 , 0.6486955649765204 ],
       [0.5590932972488663 , 0.2564647590274203 , 0.6486955649765204 ]]), np.array([[ 1,  0, 13],
       [ 1, 13, 14],
       [ 1, 14, 15],
       [ 1, 15,  1],
       [ 8,  5, 11],
       [ 8, 11,  8],
       [ 4,  3, 12],
       [ 4, 12, 17],
       [ 4, 17, 16],
       [ 4, 16,  4],
       [10,  7, 17],
       [10, 17, 10],
       [ 0,  9, 13],
       [ 0, 13,  0],
       [ 3,  2, 14],
       [ 3, 14, 13],
       [ 3, 13, 12],
       [ 3, 12,  3],
       [ 7,  6, 15],
       [ 7, 15, 16],
       [ 7, 16, 17],
       [ 7, 17,  7],
       [ 6,  1, 15],
       [ 6, 15,  6],
       [ 2,  4, 16],
       [ 2, 16, 15],
       [ 2, 15, 14],
       [ 2, 14,  2],
       [ 9, 10, 17],
       [ 9, 17, 12],
       [ 9, 12, 13],
       [ 9, 13,  9]]))]

def example_MC_13_contour_12_vts_case_0():
    return [x - 0.0293 for x in [0.546912886195662,	-0.421103532406922,	-0.643375084081520,	0.855507421818445,	-0.260686312588506,	0.206413666735986,	0.237274227130530,	-0.183297728364877]]

class MC13Contour12Vts0(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_MC_13_contour_12_vts_case_0()
        self.solutions = [(np.array([[0.5347149864904991 , 0.                 , 0.                 ],
       [0.                 , 0.4348635784961564 , 0.                 ],
       [0.                 , 0.                 , 0.6409279342710131 ],
       [0.6208227904609989 , 0.                 , 1.                 ],
       [0.                 , 0.5823479763117874 , 1.                 ],
       [0.4487843986661448 , 1.                 , 0.                 ],
       [0.                 , 1.                 , 0.7638399025779147 ],
       [0.4945033172398516 , 1.                 , 1.                 ],
       [1.                 , 0.35281189693395804, 0.                 ],
       [1.                 , 0.                 , 0.7177548806982564 ],
       [1.                 , 0.45447392342773657, 1.                 ],
       [1.                 , 1.                 , 0.7953439792560144 ],
       [0.42694898412744736, 0.7611878647317339 , 0.535306266384627  ],
       [0.42694898412744736, 0.7611878647317339 , 0.8134872139897696 ],
       [0.698776732774988  , 0.7611878647317339 , 0.8134872139897696 ],
       [0.698776732774988  , 0.29767092587247596, 0.8134872139897696 ],
       [0.698776732774988  , 0.29767092587247596, 0.535306266384627  ],
       [0.42694898412744736, 0.29767092587247596, 0.535306266384627  ]]), np.array([[ 8,  0,  2],
       [ 8,  2, 17],
       [ 8, 17, 16],
       [ 8, 16,  8],
       [ 1,  5, 12],
       [ 1, 12, 17],
       [ 1, 17,  2],
       [ 1,  2,  1],
       [10,  3, 15],
       [10, 15, 14],
       [10, 14, 11],
       [10, 11, 10],
       [ 4,  7, 13],
       [ 4, 13,  6],
       [ 4,  6,  4],
       [ 3,  9, 16],
       [ 3, 16, 15],
       [ 3, 15,  3],
       [ 5,  6, 12],
       [ 5, 12,  5],
       [ 7, 11, 14],
       [ 7, 14,  7],
       [ 9,  8, 16],
       [ 9, 16,  9],
       [17, 12, 13],
       [17, 13, 14],
       [17, 14, 15],
       [17, 15, 16],
       [17, 16, 17],
       [12,  6, 13],
       [12, 13, 12],
       [13,  7, 14],
       [13, 14, 13]]))]

def example_MC_13_contour_12_vts_case_1():
    return [x - 1007.4 for x in [1069, 843, 950, 1133, 958, 1029, 1198, 946]]

class MC13Contour12Vts1(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_MC_13_contour_12_vts_case_1()
        self.solutions = [(np.array([[0.272566371681416  , 0.                 , 0.                 ],
       [0.                 , 0.5176470588235296 , 0.                 ],
       [0.                 , 0.                 , 0.5549549549549552 ],
       [0.6957746478873236 , 0.                 , 1.                 ],
       [0.                 , 0.20583333333333323, 1.                 ],
       [0.31366120218579224, 1.                 , 0.                 ],
       [0.                 , 1.                 , 0.23145161290322572],
       [0.7563492063492064 , 1.                 , 1.                 ],
       [1.                 , 0.5668965517241379 , 0.                 ],
       [1.                 , 0.                 , 0.8838709677419354 ],
       [1.                 , 0.26024096385542195, 1.                 ],
       [1.                 , 1.                 , 0.6716577540106953 ],
       [0.7723913893422811 , 0.573670999181175  , 0.95               ],
       [0.7723913893422811 , 0.573670999181175  , 0.07021587307960425],
       [0.2622818288837936 , 0.573670999181175  , 0.07021587307960425],
       [0.2622818288837936 , 0.19820583219032303, 0.07021587307960425],
       [0.2622818288837936 , 0.19820583219032303, 0.95               ],
       [0.7723913893422811 , 0.19820583219032303, 0.95               ]]), np.array([[ 1,  0, 15],
       [ 1, 15, 14],
       [ 1, 14,  1],
       [ 8,  5, 14],
       [ 8, 14, 13],
       [ 8, 13,  8],
       [ 4,  3, 17],
       [ 4, 17, 16],
       [ 4, 16,  4],
       [10,  7, 12],
       [10, 12, 10],
       [ 0,  2,  4],
       [ 0,  4, 16],
       [ 0, 16, 15],
       [ 0, 15,  0],
       [ 3,  9, 10],
       [ 3, 10, 17],
       [ 3, 17,  3],
       [ 5,  6,  1],
       [ 5,  1, 14],
       [ 5, 14,  5],
       [ 7, 11,  8],
       [ 7,  8, 13],
       [ 7, 13, 12],
       [ 7, 12,  7],
       [17, 12, 13],
       [17, 13, 14],
       [17, 14, 15],
       [17, 15, 16],
       [17, 16, 17],
       [12, 17, 10],
       [12, 10, 12]]))]

def example_edge_tunnel():
    return [0.3, 0.4, 0.7, -0.4, -0.7, 0, 0.4, 0]

class EdgeTunnel(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_edge_tunnel()
        self.solutions = []

def example_singular_tunnel():
    # return [x - 1233.6 for x in [1763.0000000000000, 1052.0000000000000, 1815.0000000000000, 1050.0000000000000, 960.00000000000000, 1325.0000000000000, 1150.0000000000000, 1260.0000000000000]]
    return [-0.5, 0.5, -0.33, 0.46, 0.5, -0.5, 0.71, -0.92]

class SingularTunnel(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_singular_tunnel()
        self.is_edge_manifold = True
        self.is_vertex_manifold = False
        self.is_orientable = True
        self.solutions = [(np.array([[0.5                , 0.                 , 0.                 ],
       [0.                 , 0.                 , 0.5                ],
       [0.5                , 0.                 , 1.                 ],
       [0.4177215189873418 , 1.                 , 0.                 ],
       [0.                 , 1.                 , 0.3173076923076923 ],
       [0.43558282208588955, 1.                 , 1.                 ],
       [1.                 , 0.                 , 0.5                ],
       [1.                 , 1.                 , 0.3333333333333333 ],
       [0.5                , 0.                 , 0.5                ],
       [0.45238095238095277, 0.6349206349206286 , 0.5                ],
       [0.45238095238095277, 0.6349206349206286 , 0.38235294117647095],
       [0.49999999999999956, 0.6349206349206286 , 0.38235294117647095],
       [0.49999999999999956, 0.05               , 0.38235294117647095],
       [0.45238095238095277, 0.05               , 0.5                ]]), np.array([[ 0,  3, 10],
       [ 0, 10, 11],
       [ 0, 11, 12],
       [ 0, 12,  0],
       [ 2,  5,  9],
       [ 2,  9, 13],
       [ 2, 13,  8],
       [ 2,  8,  2],
       [ 1,  8, 13],
       [ 1, 13,  1],
       [ 8,  0, 12],
       [ 8, 12,  8],
       [ 3,  4, 10],
       [ 3, 10,  3],
       [ 5,  7, 11],
       [ 5, 11, 10],
       [ 5, 10,  9],
       [ 5,  9,  5],
       [ 4,  1, 13],
       [ 4, 13,  9],
       [ 4,  9, 10],
       [ 4, 10,  4],
       [ 7,  6,  8],
       [ 7,  8, 12],
       [ 7, 12, 11],
       [ 7, 11,  7]]))]

def example_singular_tunnel_2():
    # return [x - -1.3064 for x in [0.682915288844703,-2.875194419559449,-1.170635467407710,-2.411309754826369,-4.002994305193689,0.820161898722680,-1.379923984120792,-0.708030413204291]]
    return [-1.2, -0.6, 1.2, 0.9, 9.7, 0.6, -9.7, -0.9]

class SingularTunnel2(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_singular_tunnel_2()
        self.solutions = []

def example_singular_tunnel_2_rotated():
    return [-1.2, 1.2, 9.7, -9.7, -0.6, 0.9, 0.6, -0.9]

class SingularTunnel2Rotated(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_singular_tunnel_2_rotated()
        self.solutions = []

def example_singular_tunnel_2_rotated_flipped():
    return [-0.6, 0.9, 0.6, -0.9, -1.2, 1.2, 9.7, -9.7]

class SingularTunnel2RotatedFlipped(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_singular_tunnel_2_rotated_flipped()
        self.solutions = []

def example_singular_tunnel_2_rotated2():
    return [-1.2, 1.2, -0.6, 0.9, 9.7, -9.7, 0.6, -0.9]

class SingularTunnel2Rotated2(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_singular_tunnel_2_rotated2()
        self.solutions = []

def example_degenerated_tunnel():
    return [0.3, -0.2, 0.6, -0.7, -1.0, 0.8, -0.8, 1.0]

class DegeneratedTunnel(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_degenerated_tunnel()
        self.solutions = [(np.array([[0.6                , 0.                 , 0.                 ],
       [0.                 , 0.                 , 0.23076923076923075],
       [0.5555555555555556 , 0.                 , 1.                 ],
       [0.46153846153846156, 1.                 , 0.                 ],
       [0.                 , 1.                 , 0.4285714285714286 ],
       [0.4444444444444445 , 1.                 , 1.                 ],
       [1.                 , 0.                 , 0.2                ],
       [1.                 , 1.                 , 0.4117647058823529 ],
       [0.5000000060670697 , 0.49999994539637466, 0.3333333395460124 ],
       [0.5000000060670697 , 0.49999994539637466, 0.33333332712065394],
       [0.4999999939329306 , 0.49999994539637466, 0.33333332712065394],
       [0.4999999939329306 , 0.5000000546036267 , 0.33333332712065394],
       [0.4999999939329306 , 0.5000000546036267 , 0.3333333395460124 ],
       [0.5000000060670697 , 0.5000000546036267 , 0.3333333395460124 ]]), np.array([[ 3,  0,  9],
       [ 3,  9, 10],
       [ 3, 10, 11],
       [ 3, 11,  3],
       [ 5,  2,  8],
       [ 5,  8, 13],
       [ 5, 13, 12],
       [ 5, 12,  5],
       [ 0,  6,  9],
       [ 0,  9,  0],
       [ 2,  1, 10],
       [ 2, 10,  9],
       [ 2,  9,  8],
       [ 2,  8,  2],
       [ 7,  3, 11],
       [ 7, 11, 12],
       [ 7, 12, 13],
       [ 7, 13,  7],
       [ 4,  5, 12],
       [ 4, 12,  4],
       [ 1,  4, 12],
       [ 1, 12, 11],
       [ 1, 11, 10],
       [ 1, 10,  1],
       [ 6,  7, 13],
       [ 6, 13,  8],
       [ 6,  8,  9],
       [ 6,  9,  6]]))]

def example_blue_lines_div_by_zero():
    return [0.4, -0.4, 0.7, -0.7, -0.7, 0.1, -0.4, 0.1]
    # return [-1.2, -7.2, 1.2, 7.2, 9.7, 0.7, -9.7, -0.7]

class BlueLinesDivByZero(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_blue_lines_div_by_zero()
        self.solutions = [(np.array([[0.5                , 0.                 , 0.                 ],
       [0.                 , 0.                 , 0.36363636363636365],
       [0.875              , 0.                 , 1.                 ],
       [0.5                , 1.                 , 0.                 ],
       [0.                 , 1.                 , 0.6363636363636362 ],
       [0.8                , 1.                 , 1.                 ],
       [1.                 , 0.                 , 0.8                ],
       [1.                 , 1.                 , 0.875              ]]), np.array([[3, 0, 1],
       [3, 1, 4],
       [3, 4, 3],
       [5, 2, 6],
       [5, 6, 7],
       [5, 7, 5]]))]

def example_intersecting_planes():
    return [-12, -91, 12, 91, 97, 9, -97, -9]

class IntersectingPlanes(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_intersecting_planes()
        self.solutions = []

def example_single_saddle_6_vts():
    return [x - 0.0387 for x in [-0.960492426392903, 0.793207329559554, 0.916735525189067, -0.422761282626275, -0.934993247757551, -0.850129305868777, -0.0367116785741896, -0.656740699156587]]

class SingleSaddle6Vts(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_single_saddle_6_vts()
        self.solutions = [(np.array([[0.5697625394548959 , 0.                 , 0.                 ],
       [0.                 , 0.5322701622628554 , 0.                 ],
       [0.6554965417357752 , 1.                 , 0.                 ],
       [0.                 , 1.                 , 0.9209062879658783 ],
       [1.                 , 0.6204990178186007 , 0.                 ],
       [1.                 , 0.                 , 0.45913132665170203],
       [0.6215632003809873 , 0.5571364060765386 , 0.0897844496996978 ]]), np.array([[0, 1, 6],
       [2, 4, 6],
       [5, 0, 6],
       [5, 6, 4],
       [3, 2, 6],
       [3, 6, 1]]))]

def example_single_saddle_6_vts_rotated():
    return [x - 0.0387 for x in [-0.960492426392903, 0.793207329559554, -0.934993247757551, -0.850129305868777, 0.916735525189067, -0.422761282626275, -0.0367116785741896, -0.656740699156587]]

class SingleSaddle6VtsRotated(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_single_saddle_6_vts_rotated()
        self.solutions = [(np.array([[0.5697625394548959 , 0.                 , 0.                 ],
       [0.                 , 0.                 , 0.5322701622628554 ],
       [0.6554965417357752 , 0.                 , 1.                 ],
       [0.                 , 0.9209062879658783 , 1.                 ],
       [1.                 , 0.45913132665170203, 0.                 ],
       [1.                 , 0.                 , 0.6204990178186007 ],
       [0.6215632003809871 , 0.08978444969969783, 0.5571364060765386 ]]), np.array([[0, 4, 5],
       [0, 5, 6],
       [0, 6, 0],
       [2, 3, 1],
       [2, 1, 6],
       [2, 6, 2],
       [1, 0, 6],
       [1, 6, 1],
       [5, 2, 6],
       [5, 6, 5]]))]

def example_single_saddle_7_vts_A():
    return [x - 9.8588 for x in [10.2967816247556,9.45145192686147,9.54753711271687,10.6482067822841,9.81494966341055,9.31168538578250,9.80950580411527,10.7451536262220]]

class SingleSaddle7VtsA(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_single_saddle_7_vts_A()
        self.solutions = [(np.array([[0.5181192922083432  , 0.                  , 0.                  ],
       [0.                  , 0.5845643414375247  , 0.                  ],
       [0.                  , 0.                  , 0.9089924701818434  ],
       [0.2827940988012458  , 1.                  , 0.                  ],
       [0.052684562203905105, 1.                  , 1.                  ],
       [1.                  , 0.3403772053171881  , 0.                  ],
       [1.                  , 0.38167194694858075 , 1.                  ],
       [0.11126433945953157 , 0.6455997092063169  , 0.7750068922867777  ],
       [0.11126433945953157 , 0.3736158590252034  , 0.7750068922867777  ]]), np.array([[5, 0, 2],
       [5, 2, 8],
       [5, 8, 6],
       [5, 6, 5],
       [1, 3, 4],
       [1, 4, 7],
       [1, 7, 1],
       [4, 6, 8],
       [4, 8, 7],
       [4, 7, 4],
       [2, 1, 7],
       [2, 7, 8],
       [2, 8, 2]]))]

def example_single_saddle_7_vts_B():
    return [x - 9.9994608191478135 for x in [9.9998593195995547,9.9993381282115549,9.9979160205452544,9.9986053863704142,9.9999374908631235,9.999424800002032,9.9983922749132219,9.999579324965488]]

class SingleSaddle7VtsB(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_single_saddle_7_vts_B()
        self.solutions = [(np.array([[0.7645952349107387 , 0.                 , 0.                 ],
       [0.                 , 0.2050638839447965 , 0.                 ],
       [0.9297449037713149 , 0.                 , 1.                 ],
       [0.                 , 0.30848226446297194, 1.                 ],
       [0.9001678004661732 , 1.                 , 1.                 ],
       [1.                 , 0.2330959669943673 , 1.                 ],
       [1.                 , 1.                 , 0.8783231116685566 ],
       [0.9245568632304576 , 0.30449284900794293, 0.95               ]]), np.array([[1, 0, 2],
       [1, 2, 7],
       [1, 7, 3],
       [1, 3, 1],
       [2, 5, 7],
       [2, 7, 2],
       [4, 3, 7],
       [4, 7, 6],
       [4, 6, 4],
       [5, 6, 7],
       [5, 7, 5]]))]

def example_single_saddle_8_vts_A():
    return [x - 0.0097 for x in [0.454797708726920,	0.801330575352402,	0.649991492712356,	-0.973974554763863,	-0.134171007607162,	-0.0844698148589140,	-0.826313795402046,	0.433391503783462]]

class SingleSaddle8VtsA(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_single_saddle_8_vts_A()
        self.solutions = [(np.array([[0.                 , 0.                 , 0.7557238549058118 ],
       [0.3942764035660864 , 1.                 , 0.                 ],
       [0.                 , 1.                 , 0.43371211758657496],
       [0.6636582349400215 , 1.                 , 1.                 ],
       [1.                 , 0.44591240228126755, 0.                 ],
       [1.                 , 0.                 , 0.893689576230093  ],
       [1.                 , 0.181843693415429  , 1.                 ],
       [1.                 , 1.                 , 0.6989471920185472 ],
       [0.7304203043346719 , 0.5612194502632015 , 0.8665042971324418 ],
       [0.7304203043346719 , 0.5612194502632015 , 0.5102714311511487 ]]), np.array([[1, 4, 9],
       [1, 9, 2],
       [1, 2, 1],
       [3, 6, 5],
       [3, 5, 8],
       [3, 8, 3],
       [5, 0, 2],
       [5, 2, 9],
       [5, 9, 8],
       [5, 8, 5],
       [7, 3, 8],
       [7, 8, 9],
       [7, 9, 7],
       [4, 7, 9],
       [4, 9, 4]]))]

def example_single_saddle_8_vts_B():
    return [x - 9.99946 for x in [9.9985934885536665,9.9998695572230147,9.9999045831713928,9.999316745478131,9.9986117521866866,9.9998754368055813,9.9999031760062458,9.9992041920402936]]

class SingleSaddle8VtsB(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_single_saddle_8_vts_B()
        self.solutions = [(np.array([[0.6790476618904941 , 0.                 , 0.                 ],
       [0.                 , 0.6609068747725477 , 0.                 ],
       [0.6712496145236214 , 0.                 , 1.                 ],
       [0.                 , 0.6568314758218236 , 1.                 ],
       [0.7563025925181021 , 1.                 , 0.                 ],
       [0.6340288587921705 , 1.                 , 1.                 ],
       [1.                 , 0.740862014611894  , 0.                 ],
       [1.                 , 0.6189050955267495 , 1.                 ],
       [0.06864702914765683, 0.6583423425335276 , 0.6327911474536608 ],
       [0.6741308204408123 , 0.6583423425335276 , 0.6327911474536608 ],
       [0.6741308204408123 , 0.11495913573303282, 0.6327911474536608 ]]), np.array([[ 0,  1,  8],
       [ 0,  8,  9],
       [ 0,  9, 10],
       [ 0, 10,  0],
       [ 4,  6,  9],
       [ 4,  9,  5],
       [ 4,  5,  4],
       [ 7,  2, 10],
       [ 7, 10,  9],
       [ 7,  9,  6],
       [ 7,  6,  7],
       [ 3,  5,  9],
       [ 3,  9,  8],
       [ 3,  8,  3],
       [ 2,  0, 10],
       [ 2, 10,  2],
       [ 1,  3,  8],
       [ 1,  8,  1]]))]

def example_single_saddle_9_vts_A():
    return [x - -1.8061 for x in [-15.6504952739285,2.90290077342601,24.5454566157887,-24.5274127623786,21.6741877710053,-4.49696327433901,-19.7891575872492,-15.5588482753161]]

class SingleSaddle9VtsA(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_single_saddle_9_vts_A()
        self.solutions = [(np.array([[0.7461919768538839 , 0.                 , 0.                 ],
       [0.                 , 0.3444226252412778 , 0.                 ],
       [0.                 , 0.                 , 0.3709179594977872 ],
       [0.8971820815340983 , 0.                 , 1.                 ],
       [0.                 , 0.566290239442314  , 1.                 ],
       [0.5369882982125477 , 1.                 , 0.                 ],
       [0.                 , 1.                 , 0.5943788412166455 ],
       [1.                 , 0.17167141627015609, 0.                 ],
       [1.                 , 0.                 , 0.636363147083529  ],
       [0.36727097509056256, 0.45773609677722676, 0.3363982710082393 ],
       [0.8349802579126582 , 0.10124516745211497, 0.3363982710082393 ],
       [0.36727097509056256, 0.10124516745211497, 0.3363982710082393 ]]), np.array([[ 0,  7,  8],
       [ 0,  8, 10],
       [ 0, 10,  0],
       [ 5,  1,  9],
       [ 5,  9,  5],
       [ 3,  4,  9],
       [ 3,  9, 11],
       [ 3, 11, 10],
       [ 3, 10,  3],
       [ 2,  0, 10],
       [ 2, 10, 11],
       [ 2, 11,  2],
       [ 8,  3, 10],
       [ 8, 10,  8],
       [ 6,  5,  9],
       [ 6,  9,  4],
       [ 6,  4,  6],
       [ 1,  2, 11],
       [ 1, 11,  9],
       [ 1,  9,  1]]))]

def example_single_saddle_9_vts_B():
    return [x - 1233.5999999999999 for x in [1763.0000000000000,1052.0000000000000,1815.0000000000000,1050.0000000000000,960.00000000000000,1325.0000000000000,1150.0000000000000,1260.0000000000000]]

class SingleSaddle9VtsB(TopologyTest._TopologyTest):  # not quite singular, even though it looks like it
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_single_saddle_9_vts_B()
        self.solutions = []

def example_MC_13_simple():
    return [x - 0.0293 for x in [0.520482995461163,-0.839814387388296,-0.467491517013617,0.937814095887345,-0.825777099007084,0.506695544835103,	0.345318915961394,-0.861107217966913]]

class MC13Simple(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_MC_13_simple()
        self.solutions = [(np.array([[0.3610850110085973 , 0.                 , 0.                 ],
       [0.                 , 0.4971616061539861 , 0.                 ],
       [0.                 , 0.                 , 0.36484999999585743],
       [0.6417220668346839 , 0.                 , 1.                 ],
       [0.                 , 0.7301511473677927 , 1.                 ],
       [0.3535113732222943 , 1.                 , 0.                 ],
       [0.                 , 1.                 , 0.6112021903991608 ],
       [0.2619463447234754 , 1.                 , 1.                 ],
       [1.                 , 0.4889179013304155 , 0.                 ],
       [1.                 , 0.                 , 0.6454570936236521 ],
       [1.                 , 0.3490236734550332 , 1.                 ],
       [1.                 , 1.                 , 0.5050327042603207 ],
       [0.3551920917197408 , 0.49121506410348004, 0.05               ]]), np.array([[ 8,  0, 12],
       [ 8, 12,  8],
       [ 1,  5, 12],
       [ 1, 12,  1],
       [10,  3,  9],
       [10,  9, 10],
       [ 4,  7,  6],
       [ 4,  6,  4],
       [ 0,  2,  1],
       [ 0,  1, 12],
       [ 0, 12,  0],
       [ 5, 11,  8],
       [ 5,  8, 12],
       [ 5, 12,  5]]))]

def example_single_singular():
    return [-1, 1, -1.1, -1, -1, -1, -1, 1]
def example_single_singular1():
    return [1, -1, 1, 1, 1, 1, 1, -1]
def example_single_singular2():
    return [-1, -1, -1, 1, -1, 1, -1, -1]
def example_single_singular3():
    return [1, 1, 1, -1, 1, -1, 1, 1]
def example_single_singular4():
    return [1, 1, -1, 1, -1, 1, 1, 1]
def example_single_singular5():
    return [-1, 1, 1, 1, 1, 1, -1, 1]

class SingleSingular(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_single_singular()
        self.is_vertex_manifold = False
        self.solutions = []

def example_double_singular():
    return [-1, -1, -0.5, 0.5, -2, 1, 0.5, -0.5]

class DoubleSingular(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_double_singular()
        self.is_vertex_manifold = False
        self.solutions = []
        self.ref_betti = [1, ]

def example_triple_singular():
    return [-1, 1, -1, -1, 1, -1, -1, 1]

class TripleSingular(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_triple_singular()
        self.is_vertex_manifold = False
        self.solutions = []

def example_singular_tunnel_open():
    return [-0.5, 0.5, -0.4, 0.46, 0.5, -0.5, 0.71, -0.92]

class SingularTunnelOpen(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_singular_tunnel_open()
        self.is_vertex_manifold = False
        self.solutions = []

def example_singular_with_saddle():
    return [-0.5, 0.5, -0.33, -0.46, 0.5, -0.5, 0.71, 0.92]

def example_singular_nice():
    return [0.5, -0.5, 0.46, -0.92, -0.5, 0.5, -0.63, 0.41]#[-0.5, 0.5, -0.63, 0.46, 0.5, -0.5, -0.71, -0.92]

class SingularWithSaddle(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_singular_with_saddle()
        self.is_vertex_manifold = False
        self.solutions = []

def example_iso_corner_connected():
    return [0, 0.5, -0.4, -0.3, -0.7, -0.6, -0.71, 0.92]

class IsoCornerConnected(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_iso_corner_connected()
        self.solutions = []

def example_iso_corner_separated():
    return [0, 0.5, 0.4, -0.3, 0.7, -0.6, -0.71, -0.92]

class IsoCornerSeparated(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_iso_corner_separated()
        self.solutions = []

def example_iso_corners_diagonal():
    return [0, 0.5, 0.4, 0.3, -0.7, 0, -0.71, -0.92]

class IsoCornersDiagonal(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_iso_corners_diagonal()
        self.solutions = []

def example_iso_corners_separated_saddle():
    return [0, 0.5, -0.4, -0.3, 0.7, 0, -0.71, -0.92]

class IsoCornersSeparatedSaddle(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_iso_corners_separated_saddle()
        self.solutions = []

def example_iso_edge():
    return [0, 0, -0.4, -0.3, 0.7, 0.5, -0.71, -0.92]

class IsoEdge(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_iso_edge()
        self.solutions = []

def example_iso_edge_separated():
    return [0, 0, 0.4, 0.3, 0.7, 0.5, -0.71, -0.92]

class IsoEdgeSeparated(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_iso_edge_separated()
        self.solutions = []

def example_iso_edge_singular():
    return [0, 0, 0.4, 0.3, -0.7, 0.5, -0.71, -0.92]

class IsoEdgeSingular(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_iso_edge_singular()
        self.solutions = []

def example_iso_edge_double_singular_simple():
    return [0, 0, -0.8, 0.3, -0.7, 0.5, -0.2, 0.1]

class IsoEdgeDoubleSingularSimple(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_iso_edge_double_singular_simple()
        self.solutions = []

def example_iso_edge_double_singular():
    return [0, 0, -0.2, 0.3, -0.7, 0.5, -0.2, -0.92]

class IsoEdgeDoubleSingular(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_iso_edge_double_singular()
        self.solutions = []

def example_iso_edge_double_singular_split():
    return [0, 0, 0.2, -0.3, -0.7, 0.5, -0.2, -0.92]

class IsoEdgeDoubleSingularSplit(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_iso_edge_double_singular_split()
        self.solutions = []

def example_iso_wedge():
    return [0, 0, -0.4, -0.3, 0, 0.5, -0.2, -0.92]

class IsoWedge(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_iso_wedge()
        self.solutions = []

def example_iso_wedge_singular():
    return [0, 0, 0.2, 0.3, 0, 0.5, -0.2, -0.92]

class IsoWedgeSingular(TopologyTest._TopologyTest):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.values = example_iso_wedge_singular()
        self.solutions = []

######### grids #############
def grid_random(size):
    grid = Grid(size, (1 / (size[0] - 1), 1 / (size[1] - 1), 1 / (size[2] - 1)))
    for idx in grid.indices():
        grid[idx] = random.random() * 2 - 1
    print(grid.values)
    return grid

def grid_random_int(size):
    grid = Grid(size, (1 / (size[0] - 1), 1 / (size[1] - 1), 1 / (size[2] - 1)))
    for idx in grid.indices():
        grid[idx] = int(random.random() * 200 - 100)
    print(grid.values)
    return grid

def grid_extrapolate(size, values):
    grid = Grid(size, (1 / (size[0] - 1), 1 / (size[1] - 1), 1 / (size[2] - 1)))
    cube = grid.cell((size[0] // 2 - 1, size[1] // 2 - 1, size[2] // 2 - 1))
    cube.values = values
    for idx in grid.indices():
        grid[idx] = cube.trilinear(cube.global_to_local(grid.position(idx)))

    # other_cube = grid.cell((size[0] // 2 - 1 + 1, size[1] // 2 - 1 - 1, size[2] // 2 - 1))
    # print(other_cube.values)
    return grid

def grid2_simple():
    grid = Grid((3, 2, 2), (1, 1, 1))
    grid[0, 0, 0] = -3
    grid[0, 0, 1] = -2
    grid[0, 1, 0] = -1
    grid[0, 1, 1] = 4

    grid[1, 0, 0] = -3
    grid[1, 0, 1] = 2
    grid[1, 1, 0] = 1
    grid[1, 1, 1] = 2

    grid[2, 0, 0] = -1
    grid[2, 0, 1] = -1
    grid[2, 1, 0] = -2
    grid[2, 1, 1] = 1
    return grid

def grid2_simple_ambiguous():
    grid = Grid((3, 2, 2), (1, 1, 1))
    grid[0, 0, 0] = -1
    grid[0, 0, 1] = 1
    grid[1, 0, 0] = 1
    grid[1, 0, 1] = -1

    grid[0, 1, 0] = -1
    grid[0, 1, 1] = 1
    grid[1, 1, 0] = 1
    grid[1, 1, 1] = -1

    # grid[0, 2, 0] = -1
    # grid[0, 2, 1] = 1
    # grid[1, 2, 0] = 1
    # grid[1, 2, 1] = -1
    return grid

def grid2_singular_with_saddle():
    grid = Grid((2, 3, 2), (1, 1, 1))
    grid[0, 0, 0] = -1
    grid[0, 0, 1] = -1
    grid[1, 0, 0] = -2
    grid[1, 0, 1] = 1

    grid[0, 1, 0] = -0.5
    grid[0, 1, 1] = 0.5
    grid[1, 1, 0] = 0.5
    grid[1, 1, 1] = -0.5

    grid[0, 2, 0] = -0.33
    grid[0, 2, 1] = -0.46
    grid[1, 2, 0] = 0.71
    grid[1, 2, 1] = 0.92
    return grid

def grid2_simple_singular():
    grid = Grid((2, 3, 2), (1, 1, 1))
    grid[0, 0, 0] = -1
    grid[0, 0, 1] = -1.3
    grid[1, 0, 0] = -1.2
    grid[1, 0, 1] = -1.1

    grid[0, 1, 0] = -1
    grid[0, 1, 1] = 1
    grid[1, 1, 0] = 1
    grid[1, 1, 1] = -1

    grid[0, 2, 0] = 1.2
    grid[0, 2, 1] = 1.3
    grid[1, 2, 0] = 1.1
    grid[1, 2, 1] = 1
    return grid

def grid2_singular_manifold():
    grid = Grid((2, 3, 2), (1, 1, 1))
    grid[0, 0, 0] = -0.5
    grid[0, 0, 1] = -0.8
    grid[1, 0, 0] = -0.7
    grid[1, 0, 1] = -0.6

    grid[0, 1, 0] = -0.1
    grid[0, 1, 1] = 0.1
    grid[1, 1, 0] = 0.1
    grid[1, 1, 1] = -0.1

    grid[0, 2, 0] = 0.7
    grid[0, 2, 1] = 0.8
    grid[1, 2, 0] = 0.6
    grid[1, 2, 1] = 0.5
    return grid

def grid2_non_manifold_singular():
    grid = Grid((2, 3, 2), (1, 1, 1))
    grid[0, 0, 0] = -1.2
    grid[0, 0, 1] = -1.3
    grid[1, 0, 0] = -1.1
    grid[1, 0, 1] = -1.4

    grid[0, 1, 0] = -1
    grid[0, 1, 1] = 1
    grid[1, 1, 0] = 1
    grid[1, 1, 1] = -1

    grid[0, 2, 0] = -0.5
    grid[0, 2, 1] = -0.4
    grid[1, 2, 0] = -0.6
    grid[1, 2, 1] = -0.6
    return grid

def grid_tunnel():
    grid = Grid((2, 2, 2), (1, 1, 1))
    grid[0, 0, 0] = 0.6
    grid[0, 0, 1] = -1.3
    grid[1, 0, 0] = -1.1
    grid[1, 0, 1] = -1.4

    grid[0, 1, 0] = -1
    grid[0, 1, 1] = 1
    grid[1, 1, 0] = 1
    grid[1, 1, 1] = -1
    return grid

def grid_random_0():
    grid = Grid((3, 3, 3), (1, 1, 1))
    grid.values = [0.9925720470728134, 0.9800643052310765, 0.00043742770156662836, -0.6329409950477409, -0.3575186931401413,
     -0.6896483892871133, 0.5918958978141522, 0.7038446114594143, 0.25737400381427444, -0.20968756939330424,
     0.13554329265989185, -0.6392480434691001, -0.28817540622648385, -0.09500438446832593, -0.21367879199997408,
     -0.05911014527585867, 0.10421323651982006, 0.9148675716465531, -0.46046762521163576, -0.44126477410025244,
     0.26683438127382964, -0.6295454311430149, -0.7207769028740323, 0.8375131832738858, -0.22259546110693806,
     0.89546801577102, -0.17816988852927218]
    return grid

def grid_random_1():
    grid = Grid((3, 3, 3), (1, 1, 1))
    grid.values = [-0.8583842088697593, -0.8194343698205067, 0.40691238078807457, -0.7393439016515784, 0.26559704790008154, -0.4547814627228217, 0.11453643983426276, -0.6439974609941934, 0.3390424986704408, -0.9043077665242505, -0.5896330141356674, -0.7030536270066305, 0.46996213049884195, 0.1805733154706055, 0.37255558672009537, -0.7588293481306241, -0.7039762662520408, -0.6475450433172316, -0.6438824116230462, -0.11809616058767047, 0.26283644134193573, -0.4916847701264411, -0.47195566423676105, 0.7922824475950454, -0.7359521826683986, -0.1955090226414935, 0.7984240429514391]
    return grid

def grid_random_2():
    grid = Grid((3, 3, 3), (1, 1, 1))
    grid.values = [-0.5499316099436451, 0.8702352386771768, 0.2756799437920525, 0.9031323516569807, 0.12419059863239901, -0.24667403758665984, -0.1880812026229839, -0.8103140303625305, -0.9519133287307766, 0.6273785279455215, -0.3375731585874602, 0.8861104447393893, -0.7796103947315793, -0.10168454409558603, -0.24318609706597472, 0.6460347859786595, 0.121524756968868, 0.6655865486566586, -0.5502956955123979, -0.27602913696189324, -0.05339749596044263, -0.6861969739224885, 0.921259296979912, -0.7648807261033506, -0.7400509715947245, -0.9361117653305331, -0.8116266475520526]
    return grid

def grid_random_3():  # mc33 has wrong tunnel
    grid = Grid((3, 3, 3), (1, 1, 1))
    grid.values = [-0.14652922085853426, -0.7804656147883411, -0.15826348293944514, 0.05311566854020433, 0.6371004536550904, -0.2746766642047165, 0.6069334467757519, 0.5047209260245622, -0.8353971784268792, 0.09993330135867518, -0.04610225679956259, -0.8909743450859291, 0.001145481802344639, 0.311177322313698, 0.5220314145622578, -0.7775419386743683, 0.6147475464561949, -0.1405324611639831, -0.7468156741199334, 0.7829850178057651, 0.9171024254639781, -0.1757677786457521, -0.9259333353713688, 0.02843248036799828, -0.6317581315915783, 0.8238785324824489, 0.18175087833953052]
    return grid

def grid_random_4():
    grid = Grid((3, 3, 3), (1, 1, 1))
    grid.values = [-0.23666913632897235, -0.45117156844438355, -0.22323289232295918, -0.974133762431856, -0.7033049761715349, -0.9774931692013764, -0.25258494676840004, -0.7122118464734704, 0.9024021790853836, -0.40720214778280606, 0.32409234730885617, 0.4792066402371429, -0.09134426215071945, -0.02633341340831219, -0.5895915005315864, 0.0614831000185585, -0.9281332481218902, -0.7483265363878517, -0.27114898657226916, 0.9731839022372035, 0.08415996348471433, 0.9624615691222993, 0.7362800854828464, 0.9968729626184751, 0.7942470881438706, -0.7770101427520852, 0.2547203957214077]
    return grid

def grid_random_5():  # mc33 bad fit
    grid = Grid((3, 3, 3), (1, 1, 1))
    grid.values = [-0.0780208414077248, 0.9344868510375441, 0.4162243620243691, -0.21977285491898324, 0.6041274457864534, -0.25488210182375126, -0.7062591892837105, -0.5876017568223495, 0.06205236109394208, 0.02458323353838998, 0.42404682774707947, 0.02403257752414567, -0.4853294882612098, 0.06329993863777084, 0.6028416543017856, -0.9368383834581626, -0.017070627570113572, -0.36280937657298784, -0.49669418290202283, 0.9671810719892249, -0.6183612364999467, 0.9084951269204584, -0.9968249546276517, -0.7767786623717268, -0.47132318455681443, 0.24283706189262477, 0.579238989202919]
    return grid

def grid_random_6():  # mc33 bad tunnel fit
    grid = Grid((3, 3, 3), (1, 1, 1))
    grid.values = [0.21544848542941586, -0.37132045602406016, -0.5448218737165467, 0.48489452261858457, 0.9038535412445987, 0.08591229374321019, 0.9036519750505758, 0.7352754806915534, -0.8731028765480733, -0.9852240349848866, 0.6334119562105496, -0.26820935817198044, -0.7899196200923297, -0.19697455624654792, -0.9376419068855866, 0.09160824900510711, -0.03670189334630791, 0.2567786867632358, 0.30902781464477713, -0.356597296663278, -0.36711279521652407, 0.0403956222598123, 0.13228457558487072, 0.44735596487587403, -0.4296747977308901, -0.40889324356855195, 0.46236463875102696]
    return grid

def grid_random_7():  # mc33 two wrong tunnels
    grid = Grid((3, 3, 3), (1, 1, 1))
    grid.values = [0.8012629491675771, 0.8321854960578663, 0.473145842741153, 0.19953293087669488, -0.12679040140635345, 0.21245844420562987, 0.4215261598468334, -0.6988902861253856, -0.8681463324880041, -0.6526832408372127, 0.523205405176514, -0.28049839533158005, -0.34543538492763703, 0.120511320148325, 0.5228375210667304, 0.5830919631050353, -0.7038506575936949, 0.5375221125173475, 0.09741161085404437, -0.6275303688717513, 0.3230613701938694, 0.5417390530732131, 0.6105334142584224, 0.09300843728670194, -0.9491490842119519, 0.6116623229038747, -0.22644906667010423]
    return grid

def grid_random_large_0():
    grid = Grid((10, 10, 10), (1 / 9, 1 / 9, 1 / 9))
    grid.values = [0.10614757773493966, 0.817431346290066, -0.23275333709269708, 0.21230226809540853, -0.8037495956920837, -0.7372607216472085, 0.87956963645787, 0.005053599462976122, 0.23250912208313967, 0.9786845885041897, -0.20547078669289687, -0.2083823037877639, 0.33402512650234395, -0.46814610311098526, -0.01129071780942037, -0.19526854810666538, -0.24556302790213125, -0.16138766409677463, -0.9744075183965304, -0.11341886084906982, -0.7118727535996423, 0.680412119718526, -0.4039632209403434, 0.41252075171468294, -0.6429860589200516, -0.6005235043755981, -0.4178596131883958, -0.1801513737392808, -0.21886066248748737, 0.45329663060795844, -0.528402241295556, 0.24759681842851777, 0.8696239789984357, -0.18205973275277842, 0.409727262521008, 0.15707561258715397, 0.7974688224808348, 0.7010472484673069, -0.32517504522257834, -0.9556967473798925, -0.7690419638536463, 0.518458665634979, 0.6772422101047011, 0.3206451843901912, 0.7165108944913456, -0.06930664329456615, 0.6216322403831467, -0.3231283636001132, 0.37534245320889337, 0.5391401627942742, -0.5088632902118349, -0.8188024209097224, -0.5978745543166946, 0.5384154253102353, 0.4193966551096704, -0.3028658301198386, 0.4837166238307462, 0.8010335798972368, -0.3547866996871549, 0.7902396149582871, 0.547407216484449, 0.8508671004345261, -0.25109817782627086, -0.10330104584953226, -0.7265993240413187, -0.6205279888549198, -0.014243719282097445, -0.09585063757926116, 0.18438876865416498, -0.27300122004723804, -0.2146856902625034, 0.44806251556164955, -0.5240857068921565, -0.42092697249865907, 0.26610511558096905, 0.7212374063172688, -0.1372202989938507, 0.4793728101973842, -0.6119675485268599, -0.5189701483880089, 0.4055723014171311, -0.4390496032523996, 0.26711460225397365, 0.9982309148319819, 0.13272957819984432, -0.5363233606740849, 0.06469617454593934, 0.9368939374657654, 0.43460664195630905, 0.7605884627500985, 0.29409130540537465, -0.13956929601174206, 0.8266091839257763, 0.730785985528827, 0.19690721950806633, 0.8010689807258962, -0.5248524222682209, 0.25990668315883103, 0.8220184333998792, 0.5595428003435525, -0.485153624738462, -0.32157300801783717, -0.7513505739083888, 0.16763890725293984, -0.9676902474380926, -0.4349328333462861, -0.5558740891380722, -0.8047896634375791, -0.43767634976812886, 0.4145615353811898, -0.774561292739804, -0.3430656258182516, 0.9788130153640375, 0.8248800272781445, -0.7927190209912491, -0.7747513528722563, 0.4577539510523261, 0.8725909029823122, -0.19019695349258314, 0.5817254554893094, -0.9940808935033776, 0.9550904113044294, -0.5620522555985084, 0.6633299649181708, -0.6828082688586419, -0.5668143225650921, 0.8685413631959258, 0.9112400022604075, -0.8089153245163987, 0.903640412918294, -0.9579103713536614, 0.042220521208674144, 0.3344286539487302, -0.875722139185489, 0.3636001654200849, 0.8952385854620484, 0.02802088416818349, -0.7846556371418563, 0.5220916524087178, 0.8108482253597344, 0.18281995317134103, -0.3209492474249662, -0.5768351055820959, 0.8264929875589964, 0.5122064348878288, -0.023821735173576775, -0.6824698271314784, 0.5847007721137807, -0.1694669458635456, 0.36368452131585927, -0.0419676137058107, 0.43580685205133096, 0.13736711086901354, -0.08781264002123246, 0.06672193035682139, -0.4883583789245589, -0.9992712136451616, 0.17831312479427486, 0.25231516462483006, 0.7473827858662951, 0.35208885977785465, -0.10360483649373209, 0.7797758402789567, 0.7026002027991767, 0.2543410143356033, 0.4319515932460678, 0.46058073906954866, 0.021413416241995664, -0.9576175023906301, -0.22133621209556997, 0.19631219107335207, 0.3552834792953239, 0.2838201318204743, -0.978948671891934, -0.012658184114423987, -0.4041151668876135, -0.6283036409901379, -0.2635638827519857, -0.8443083339791608, 0.9987120257195128, 0.7873776280051972, -0.7166566050873837, -0.701100449002833, 0.554576489225862, 0.36449785974665416, 0.7784275154629765, 0.25702251707367973, -0.7929287447878375, -0.0922235399886373, 0.22868373767534544, -0.16822067395187879, 0.17440593275091487, 0.41213742403836995, 0.21114666666769666, 0.11048867278879748, -0.6731236281382018, 0.3129744720239047, -0.4489595445824863, -0.8089244184392084, 0.9303452329215738, -0.8170363626943613, -0.7637967526021894, -0.8058917466429811, 0.640006859470059, 0.8939200143281494, -0.7279700209871283, -0.504406204729672, 0.19139915210410763, -0.5621804727634048, -0.8001613226835749, 0.9139537397123956, -0.49480694513280765, -0.3322392564429404, -0.5789777608141404, 0.13324778540684945, 0.7012000598754886, 0.2779379899362069, 0.11415151810958135, -0.8146749035548473, -0.4781586462229048, -0.9478050115525671, -0.9569919457747336, -0.022706471591903865, 0.13086103306383645, -0.4160947791905363, 0.28969695495218395, -0.5753605140315703, 0.9394337824214298, -0.04016611424722827, -0.2467016951855152, -0.9425667851151709, -0.49340073770626725, 0.021718408087028962, -0.9766588192970258, -0.8394142945956327, -0.5661711682710018, 0.3753580668435932, -0.3062910782804855, 0.9949247495786102, 0.1534566744500614, -0.9967331642120809, 0.0537054182587664, 0.16277950899291826, -0.8244255753007981, 0.0034861079551036234, -0.18637383047495737, 0.11523982391836451, -0.7906623969375501, 0.01591189405419735, 0.6885007989614251, -0.5514548341478913, 0.22001185596389616, -0.748296731629742, 0.4836409809003084, -0.07220924669707185, -0.5058844342557154, -0.016075219831006127, -0.4050207932536165, -0.3941125016866185, 0.16564625720615545, 0.1078994534049531, 0.7496882252124557, -0.6177794432586134, 0.9266563250144144, -0.8273324937202582, -0.976239795782168, -0.3857854781686223, 0.02955850734784038, 0.13688929216891754, -0.8641979397837758, -0.7170530331953879, 0.5808153093726323, 0.6095541167883951, -0.7835434190956014, -0.5828056614971169, 0.39015238867405055, 0.28972808141788975, 0.6586200244310854, -0.14291630595469185, 0.8963851608195315, -0.5914046304361278, -0.3798609057161728, -0.07889666135449813, -0.07028467671934302, 0.4616208808857154, -0.7884749161567695, -0.6592955734477404, 0.3233157083468492, -0.9625054978297167, -0.5007525603418335, -0.6673573675969622, -0.9491935413686943, -0.1706585550090025, -0.4173319755878626, 0.027467587467212118, -0.9181640584126896, 0.9285201020578153, -0.08071830352609011, 0.3114531529997897, -0.9506751106713303, 0.13541608989571108, 0.3539112770636996, -0.19424826586760946, -0.9670278394907093, 0.9600675780702053, -0.5311349196058719, -0.8218918086650924, -0.25750011145041163, -0.21666889711706117, -0.916178052380495, -0.39553006458437534, -0.20697718313184477, -0.055946455884312796, 0.4414184938614394, 0.006540198370646344, -0.36056578906775494, -0.8783274885727994, -0.19176054806878917, 0.9394367996268853, 0.7997776558435126, -0.9334295573223028, -0.5613326718735265, -0.8187589927774559, 0.8574168458943259, 0.5816142264809772, -0.8470186592486806, -0.2173992778830327, 0.8244833078936082, -0.49470292419981465, -0.9400180203315132, -0.7563398850177128, 0.8366134345818319, -0.21459797490565946, 0.11093142962799796, -0.042477928442426816, -0.49014888947886615, 0.6148940234376552, 0.31079862084653853, -0.1454364725468975, -0.8930034062735883, -0.40778299539708507, 0.9465027068752976, 0.8817413393739741, -0.7641065500748376, -0.3489316404751339, 0.4457370811768748, -0.7042582480082051, 0.2311916223379631, -0.6728156835668575, 0.6983446214519964, 0.4876453126459481, 0.6210020037988944, 0.3364306430695856, 0.8204598610654119, 0.036723239785073414, 0.10098662772736255, 0.7219672991024204, -0.6412956614931424, 0.08127132285037053, -0.5660725288805126, -0.6890187215869408, -0.7338620398899123, 0.37534032744450996, 0.6537832039467362, 0.8515910383572616, -0.6963208211623106, -0.5600684877943214, -0.2578621035409301, 0.4380598767811008, -0.7969382822359481, -0.5698266003975161, 0.9014181257596734, -0.28235932153851495, -0.15231621448433352, 0.21328066707368465, 0.7102082056072772, 0.6494909573501464, 0.13102350574083776, -0.6660225585533344, -0.7632056457538958, -0.36100771139829835, -0.5951584523764661, -0.7948044605502855, 0.016892031802624663, -0.6389314393816843, 0.4983600230554972, -0.03617953952727837, 0.8018503160870134, 0.5670836253570768, 0.9526831844600727, 0.7811197685250686, 0.03502344208168151, 0.30365345023098644, 0.14009627806056835, -0.38283737729706635, -0.026440143495064294, -0.8733816379091608, -0.6454287962669438, 0.7368068756247823, -0.9375152954334547, 0.7527726056671566, 0.6128817651810581, 0.1943790350383101, 0.9893355475782446, 0.08319387192124017, 0.8148615520708586, 0.2294254535141509, -0.749387351453298, 0.06116662826796704, -0.7678909581072457, 0.7196342206967206, -0.7739188245390465, -0.7399590591648544, 0.9899292528400652, 0.5054295951713448, 0.3341784320309491, -0.8616154774841396, 0.6265084516859278, -0.28924893379060523, 0.13241923043683967, 0.9302701065739569, -0.1684996178443401, -0.5254659965818085, 0.4763867757362954, -0.4522241478803777, 0.6437879000190563, 0.4085848702925665, -0.5321608642906914, 0.9010230026792052, -0.4424298314124717, 0.8676178933313969, 0.27371923538462073, -0.426939891827687, -0.28018725007961565, -0.31879892832060674, -0.23656075055191006, 0.538879282186286, 0.42280542715709624, 0.8770817920151113, -0.6280478172393793, -0.2024039555706807, -0.8481883536178485, 0.3615380695378074, -0.07547885294965728, -0.2289523689324391, -0.1120492922549936, -0.5211269670587211, -0.21397675618035628, 0.651939591918951, 0.2575615694449369, 0.4621030712048033, -0.5503411936718265, 0.3831377987881244, 0.21227683170776834, 0.6102185906513455, 0.838260297261588, 0.6289515323643071, 0.21131041415730079, -0.7332091774580922, -0.24868612921916555, 0.2606634496025506, 0.27517669011301393, -0.5616440253605914, -0.06337314485555501, -0.017275651659572677, 0.35221821228662376, -0.28935014648078416, -0.6802124469601858, 0.41636207667732705, 0.25189115117402294, -0.11527415205049785, -0.2746273679909734, -0.9771733420332935, 0.48233047241795557, -0.2158475082399287, -0.6208828069816978, -0.5070591629109367, -0.7470104859367452, -0.4897089992062029, -0.6190235259304271, -0.9075369027910092, 0.5648031891502567, 0.6110470215768014, 0.7847340796455231, -0.46568295177044017, 0.520146941800292, 0.3061992431910272, 0.629257582964228, -0.5684666971017915, 0.7293151250745771, 0.48894656674935666, -0.8843887700278301, 0.8683836534022484, 0.928773238446666, 0.9768976216484679, 0.6890387325547256, 0.3500530694331303, 0.9676880682186555, -0.40639711048039984, 0.024744508948019828, 0.30460243067424275, -0.795792426759903, 0.910556253263179, -0.2101652842315811, -0.0972199176349069, -0.19859539094382206, -0.9684018562152297, 0.09282239989735697, -0.5724626507942996, -0.20045171374742732, -0.5554332107501323, 0.3547993777210643, 0.670044844913485, -0.597788681446699, -0.014695974159500524, 0.34288693681540194, 0.02231560102105301, 0.194577220415054, -0.8376417544044796, -0.13719841952519296, -0.30827934900945553, 0.822718256174318, 0.20536331587466616, 0.8301325196431497, 0.8170582657683685, -0.9039921413026797, 0.5552099150323633, 0.8403714076143249, -0.88730231682917, -0.9161830189608056, -0.4135966980350423, 0.08166066307550057, 0.24849332662624213, 0.3634935042743195, -0.7429086228056034, -0.586245260786034, -0.1550190530338622, 0.9552274490818484, 0.2735307238980089, 0.12728861242526524, -0.7703069294494833, -0.6668477868426119, -0.9428045778973344, -0.04395423382044239, -0.6784083826569511, -0.449579685678378, 0.4105747073950985, -0.5829675572067188, 0.7316286763360755, 0.6954744257129144, 0.3012458891417087, 0.35660922936018524, -0.21191382716080676, -0.19687510595179392, 0.37319076946698027, 0.5095006340278079, -0.5644279706453494, 0.2899057877512159, -0.7900056323834035, 0.12517843510729398, -0.6994657941549083, 0.6122104432411555, 0.0563147577961689, 0.5101151923435396, -0.21620189558614933, -0.4196730774103059, -0.6641528735641724, -0.033455714418016314, -0.2609502949524558, 0.041758436020370215, 0.26401984792438316, 0.504822182149929, -0.9946985400208372, -0.21083891406088529, 0.9709195196534146, 0.16031849383493113, 0.4790714900682742, 0.7992070805543574, -0.7432243749821428, 0.9698738287788007, -0.8956026495775924, 0.5329942875460227, -0.7564879860694846, -0.5411970355697362, 0.8640761734231701, 0.7426212751988193, 0.006603641871414823, -0.5926334655654792, -0.7853706385025039, -0.4794440134145159, -0.9692723069150293, 0.3412636810983094, 0.48346000950197565, 0.2644886338320491, -0.8884482906249258, 0.7670314728682623, 0.6868457800363053, 0.6770302311968321, 0.39107317081597137, 0.8271484928324342, 0.449888172979346, 0.05695211749428686, 0.9034976211057679, 0.8771266266584081, -0.5794171263688148, -0.5174956648605933, -0.8300319009923649, 0.05474947425311627, 0.5203400078816518, 0.08447949163505752, 0.23344341531206036, 0.23482007839634722, 0.08686507307427327, -0.21067471943710347, -0.15553999208864466, -0.5819415671910786, -0.7110331683217448, -0.5897306478739108, -0.48993515056012904, -0.3927571091599589, -0.23811205590109874, -0.9604218503670232, -0.5328268782184402, -0.08693292893113536, 0.7182826769462662, -0.6444973041771507, -0.5227557295439658, 0.20533102417384264, -0.743387814504914, 0.7119348476469662, -0.03238254777337546, 0.9178812163233814, 0.3432787156028667, -0.34555927300439726, 0.4779842386232114, 0.17002266177725556, 0.985287307918058, 0.8634328844267014, -0.9844683626517738, -0.40314301700812827, 0.6147939011170891, 0.0784336325451227, -0.5933085325877492, 0.8849220373519671, -0.21294078627655733, 0.4985938592395611, -0.7752488015018939, 0.08007898947948555, 0.1964169871838095, 0.9885002699365195, -0.394443975795562, 0.46305790969695715, -0.6273895478508797, 0.08865039954718834, -0.28064339251797565, 0.47197829116884105, 0.45961440477789406, -0.413852477506947, 0.9820360525220759, -0.053400418089793567, -0.9175621846938742, 0.020323142227560043, 0.5060776605958499, -0.5530648828790372, -0.9066970954428453, -0.24864063809663284, -0.4223100584980637, 0.39913392112249624, -0.8065689952541515, 0.7634862914755058, -0.6014319506750532, 0.6425347285874943, 0.33909496467187683, -0.4249664274809246, 0.2195886621817078, 0.28065090588206876, 0.25281501925034244, 0.6274182534791206, -0.12622760373460773, 0.34239535660854603, -0.3590564999565251, -0.4717772029734808, -0.5690134531669058, 0.3886795840905226, -0.6466789927201986, -0.10090848936775942, -0.03353993715376302, -0.13554735164436216, 0.5825601183724838, 0.9166648711669596, 0.08219656289021438, -0.018257548284327463, -0.07694462731038021, 0.7116822645917038, 0.4498092825891813, -0.19071019259271504, 0.40759135359150345, -0.8198161233320269, 0.9370488573412377, -0.6839898654299492, -0.32328798225945055, 0.3153529335901728, -0.9049321842216631, -0.14796960172344154, -0.7091344109109223, 0.971089824873731, -0.6628350955754092, -0.7674021391210635, -0.14984558402232184, 0.926506232451864, 0.4297675343568119, 0.3882794365403168, -0.42750178177465314, 0.3333827505839204, -0.04578246634081706, -0.580726908963306, -0.11414030458555247, -0.4406812426215476, -0.6631615351537918, 0.545487842634071, 0.9397202956866353, -0.6841046056229152, 0.43677642611124545, 0.3181388850158855, 0.7292533259375262, -0.31251924212494675, 0.002828967039408159, 0.2991470979830515, -0.5626109921524092, -0.02998768920217043, 0.9549068268363134, 0.7700371984058658, -0.7168520005816781, 0.8703643332751514, 0.8926729299520455, -0.16374144663824652, -0.008760574653724262, -0.8761092636201513, 0.23413894880260333, -0.7148982873278737, -0.8000596026544009, -0.6197522969067539, -0.5905398754854896, 0.593376758321656, -0.17764018346641985, 0.3794746906758182, -0.4552387988417057, 0.9069214508840819, 0.41216569698844663, 0.41883078669581875, -0.4772105235659103, 0.9120839113876782, -0.13072511816944843, -0.7652830417143903, -0.16816247715894184, 0.417772288343381, 0.4024878530328746, 0.012743925473286177, -0.3991138965082479, -0.05985559020755815, 0.9697434107036667, 0.7764027207978355, 0.9637958618595357, 0.2858769846583762, 0.006490665524326644, 0.7006793797116972, -0.006456373133043591, 0.19680003686931857, 0.36549695489441536, 0.5269739575189611, -0.6914357677340957, -0.3973631611184507, 0.4341200387750963, -0.9503555943932651, 0.5131548767879635, -0.8803078745436888, 0.8616591974442271, 0.17845167494519987, 0.844459838968046, 0.10541401109924453, -0.9427160185208052, -0.6833807235656699, -0.2086110799603551, 0.4927624472484955, 0.631923321126618, 0.1439269669929979, -0.8417264699716871, 0.0728794767969938, 0.3151392367271748, 0.17524103838742033, -0.8077514551942748, -0.22512207644449944, 0.7222827633347007, 0.4429277714629223, -0.6899377475427151, 0.165659934465767, 0.35203003580751324, -0.9290342262517053, 0.1421901817654463, -0.009423183463684204, -0.9053403149087469, 0.4082525439119986, -0.5911281444351011, -0.1311997571369532, 0.5274710241334293, -0.3252076004007698, 0.47337719671147993, -0.06103904462122167, 0.5120418607093984, -0.6741959542140139, -0.8794770050364835, 0.3818552995609381, 0.17046123334662533, -0.43728578416355157, 0.6858509325656841, 0.6751154805329864, 0.5043609445906205, -0.5260267606028994, 0.9659445954511883, -0.5470220901602834, 0.07701960442237232, 0.9817938998934637, 0.6549387589646163, 0.04954810548506461, -0.4519886134591218, -0.5583724508132974, 0.3789021508596666, -0.15454047022099093, -0.3662170334443382, -0.33498503535599067, -0.5674152628127869, 0.6564243614596834, 0.3567938080387365, -0.7275003758364416, -0.8048539226848426, 0.3671156242377962, 0.832721852608211, 0.5183202237514151, -0.4996467840426926, 0.24623000211428936, -0.07286339339211056, -0.6046307132453408, -0.7123676614315042, -0.12426731342907815, 0.6452358815926484, -0.7360195653421626, 0.2065772489417934, -0.9791486903155828, -0.3187599138116082, 0.5125853455599851, 0.6285924980252064, 0.12184440147092723, 0.29157026471649417, 0.6022802252767998, -0.5417043282922425, 0.7806417417925333, 0.7054346192033913, -0.4274850613787775, 0.9561113880110603, 0.4990605213133239, -0.9063326928841084, 0.7073890470900401, -0.8413208179539933, -0.26502638934472245, 0.07756719811569801, -0.9656808999210984, 0.006843985537965613, 0.1679278190315241, 0.27814164469548164, 0.5987661186526034, -0.01896340116888373, 0.8769036096511245, -0.5483911401902282, -0.4336362558505822, -0.8375209793438991, -0.8054606304333094, 0.8852389335518211, 0.038543914201930374, 0.6476369487010423, 0.44694875529592104, 0.7043657777947918, 0.33509911132819403, 0.49144732261276625, 0.7763773619302834, 0.29434931422841126, 0.7234492192810618, 0.8244583955052212, -0.32395190362485016, 0.48492016607515676, -0.5436769644023935, -0.5646111937504581, 0.10698286908103172, 0.7996776396662706, -0.15499308227585185, -0.2001787374659787, -0.47463093326133365, -0.2851879443468759, 0.7052265442979253, 0.18278479058324182, 0.29414216600191656, -0.3311639558053432, 0.12601567216221676, 0.6588380666901787, 0.4410367734770493, 0.26863531738931923, 0.25305390467079825, 0.49294990742922984, -0.50394887860292, 0.9843471090015627, -0.48951432713222265, 0.3596173257344075, -0.47927832945208193, -0.5930674004412528, 0.12489334333338742, 0.03975934187206498, 0.2952609143953111, 0.8613677701165079, -0.1311366130062659, 0.680399237725499, -0.7523973286837078, 0.17456172369345224, 0.0022559143532603354, -0.3551879267794671, 0.4965179892514444, -0.5739187471426523, 0.255335272454851, -0.27711603341586843, -0.46804195443263197, 0.7407332338692669, 0.6552578063848076, 0.07111227924375574, -0.5195791968499797, 0.38979242963785143, 0.8707171147278061, 0.9586682459700429, 0.32701919053784545, -0.39002570809222115, 0.11271819552992679, 0.745497485625966, -0.6792678352419976, -0.756617921460301, 0.4853385226679723, -0.9236582764834831, 0.3397955183588639, -0.08554001085254592, -0.9969184924022947, 0.441479217051691, 0.7495030515510439, -0.6855537702531538, 0.9322462897187542, -0.22397643428983738, 0.07210679356464134, 0.13080766534647248, 0.06058383086935892, 0.17343909455097406, -0.9365832758520374, 0.6363229082659825, 0.6528784773140883, -0.8844241209008572, 0.9876748834110067, -0.7889837286062773, -0.08752488135044456, -0.1427270105778018, -0.6425520168467125, -0.9098303149278428, -0.34144192338713686, -0.2236433860126732, 0.08336852862616762, 0.019865277309981222, -0.16632746751643346, -0.6899666558724593, 0.3180321484870443, -0.7688195559600084, 0.24048749118984847, 0.8524498839500316, -0.023563897181858362, 0.15823562907994404, 0.040575884997522405, 0.22675163922228614, 0.8136452975063173, 0.10419985932520559, -0.26666565127994146, -0.5618636420464251, -0.9968539950054536, -0.43429192825503504, -0.42211218224517477, -0.9639081139927295, 0.740116051899276, -0.9367585058220358, 0.7100017560647061, -0.02461139359564557, 0.7025334398842229, -0.9633803812550472, 0.09765436914537351, -0.6368935749945499, 0.287520471274733, -0.242484617901765, 0.4730451909225366, 0.4791968832385607, -0.613971235453995]
    grid_empty = Grid((10, 10, 10), (1 / 9, 1 / 9, 1 / 9))
    for idx in grid_empty.indices():
        grid_empty[idx] = 2
    grid_empty[0, 3, 5] = grid[0, 3, 5]
    grid_empty[0, 3, 6] = grid[0, 3, 6]
    grid_empty[0, 4, 5] = grid[0, 4, 5]
    grid_empty[0, 4, 6] = grid[0, 4, 6]
    grid_empty[1, 3, 5] = grid[1, 3, 5]
    grid_empty[1, 3, 6] = grid[1, 3, 6]
    grid_empty[1, 4, 5] = grid[1, 4, 5]
    grid_empty[1, 4, 6] = grid[1, 4, 6]
    return grid


# [-0.39207868618487063, 0.05748318674566888, -0.8741643956941354, -0.8064549582722333, -0.24072712662785611, -0.2339944156545124, 0.5268163532271932, -0.6963671360072397, -0.8824216136304639, -0.9562684447312741, -0.08440884235667867, 0.4545805702138297, -0.11614771059748974, 0.6796916174755405, 0.9750581844160731, 0.4099116816348407, -0.9339244354709504, 0.26986571371792767, -0.47553330372311176, 0.7723447836306265, 0.7765801849582332, -0.8767796055891097, -0.08361525909367873, -0.4411852183008529, 0.01338704049562267, 0.7198932323611023, -0.9450047947133853, 0.48657992695528507, 0.048394495516536296, 0.327535142773278, -0.12015946803131627, 0.30615103255102305, -0.14632576406807463, 0.024558134633460016, -0.3955809079110957, -0.7070193299514327, -0.6321660787294159, 0.12178495231754449, -0.9700701037957751, 0.006110773792968427, -0.0733726069078815, 0.8979907205878663, 0.5946906215784413, 0.5507672558349308, 0.3663585723210112, -0.39485172786668965, 0.15562081174647013, -0.6569786039753771, 0.571598479165206, -0.9867282338335519, 0.8515414365259495, 0.15949184852878662, -0.9790228937347587, -0.15639025789873773, -0.5130741303048756, -0.582888900765534, -0.4988027963767889, -0.6490927452333308, -0.6992337391536187, -0.8931551285931008, -0.9262476055386966, -0.8549168525698327, 0.455854820144886, -0.4799101170623692, -0.3560594714415839, 0.7425647754269225, -0.9777830768238116, -0.8832388265223519, -0.9228263635419862, -0.5111211635964212, 0.8010605891924463, 0.3182269736118557, -0.7827824100400838, 0.5975090866598498, 0.2175851955882686, 0.6987016140627658, 0.4047075442082775, -0.25850791884662705, 0.5107880669974934, -0.04437913614757716, 0.47051981909502416, -0.3927434827003513, 0.9304523498277348, -0.0533857457925524, 0.41623732060304874, -0.32561112536268966, 0.3130608686118157, 0.9870192673295777, -0.6958342109216076, -0.3992715625025389, 0.8493543319735255, -0.7255921710306041, -0.40220466520128095, -0.2669630389081836, -0.8948454667477941, 0.08283998713110119, -0.3759526994121869, 0.7495914104432078, 0.6857910085794072, -0.4783969407379476, -0.8216940152638215, -0.24373753998382885, -0.553894002836971, -0.5224215755772708, 0.6603306016257093, 0.4445780272407416, 0.9054350463347873, -0.47624617304024786, 0.1804481076433162, 0.961731662211254, -0.7866184575278274, 0.06735078134719785, -0.4964273835105042, 0.8335551812541022, -0.22674683823833863, 0.47378889962726056, 0.1911618171276226, 0.4507292651300703, -0.016893927654827445, 0.8730758159640835, 0.4583832590645349, -0.5288157228268073, -0.1396604664102543, -0.9254794939000643, 0.7505823299295149, -0.18506383859056985, 0.35358366789413176, -0.7067691954357169, -0.40120451401263324, 0.6887142459912381, 0.16897136335246987, 0.4648727548222631, -0.863180141253566, 0.5179422200050574, -0.9840153765194675, -0.8836524580495355, -0.928897446272289, -0.28304259670564313, 0.1087062658364153, -0.6378078479914637, 0.24223567363284526, 0.7143892454070504, 0.2663080917794074, 0.2486869217222114, 0.7075587734985964, 0.09216961671569912, 0.6873193644613229, -0.9360366042188162, 0.659854587626123, 0.08627635527867716, -0.47104869840169195, -0.23362755313225758, -0.6896084552643269, -0.9840176616077789, 0.5730754507198146, 0.9694161876155987, -0.08181606038373168, -0.2986506205185131, 0.4294547724029385, 0.8894046165694462, -0.657064787380409, 0.7175818998281793, 0.12015358132608944, -0.5429466605315856, 0.8281141969082033, -0.8628367383777413, 0.3240317314353951, -0.4619737560676962, -0.34651394366864574, 0.5283754505800806, -0.9736019134386147, 0.29568155132409824, -0.9478216712706182, -0.1406442061857991, -0.7228043406666913, -0.8047942683089968, 0.04859459702497926, 0.35881337501279975, -0.533834476805523, 0.7809899971196812, -0.8341477459960631, -0.8410276972410595, 0.5410716144254433, 0.7070369179716587, 0.6956369036704892, 0.5909418915064439, 0.07719204531294865, -0.13237715508976433, -0.31383357977981086, 0.035831794273218254, -0.5900216921895161, -0.545502843963497, 0.9028214362765217, 0.20550674441634675, 0.7443749181008339, 0.34626667877969575, -0.8690314587191934, -0.8360773899211569, -0.5262822602258264, 0.1755555967075193, -0.6987425907393647, -0.7766119209274487, -0.35418403964000267, 0.323548046244426, -0.1782624492299909, 0.2465117127365266, 0.3228867764222083, 0.4402131678457919, -0.39882962908564323, -0.5076488068594156, -0.17129958153261815, 0.48944254591674485, -0.01197596618889274, 0.734577890703694, 0.3505879440141386, -0.28289955478008544, 0.9910751590664764, 0.04829565518436252, 0.3071544042339651, -0.9070702444172156, 0.6887581324582941, -0.08708715425574987, -0.43986802224489163, 0.34272529131481844, 0.680334857970273, 0.9640838119495465, -0.5615493698657745, -0.5127450693966942, 0.9393777716683132, 0.655975104887297, 0.13816159126787886, 0.7333807419328509, 0.6632311778235076, 0.3710940938158651, -0.2276687936055941, -0.9139952245967153, 0.30681203692091885, 0.3362013055305806, -0.9596532865071239, -0.7716126919548236, 0.3219342898741149, -0.7577395182135778, -0.48806322449380435, 0.9147762977321552, -0.11234988860501227, -0.08967925791707243, -0.3907399330244883, -0.4447016028586197, -0.5853846222449643, 0.9271698688716532, -0.3387224905894144, 0.619934469352148, -0.5464056072860957, -0.8058554425899593, 0.27188953204547883, -0.2003664236135716, -0.898938830675525, 0.06997756555205781, -0.07520782937321768, 0.9385053997158661, 0.05783244195517967, 0.9620456178514174, 0.8273040581700941, -0.07614341856842821, -0.9962046149055854, 0.9595737323342268, -0.9366757142060593, 0.7093828252352121, 0.6549354065826118, 0.29179360813156996, 0.8445949434481992, 0.583124820315158, -0.19675387986241333, -0.1973786317144679, 0.5097334621363996, -0.494966296977003, 0.8093332535783089, 0.7300723519285472, 0.9868728688044592, 0.7249005329197036, 0.9490078053682423, 0.987501212230441, 0.2640131332347784, 0.5055365605941351, -0.2154972665490904, 0.9218350808802345, -0.3677049833509125, 0.2116054670090768, -0.0438096987975114, -0.37187853583220876, -0.4166053261404805, 0.5306754625419181, -0.20219910494021764, 0.6345970250825081, -0.5836405827987394, -0.13551714567861795, 0.04127843586000157, -0.1093832580341465, 0.43311203047337, 0.6531219321923669, 0.45191002472603836, 0.696117375255217, 0.40510618492352735, -0.568601085712712, 0.8629277883168263, -0.5046742598056959, -0.04960054498477251, 0.36766392676010584, -0.9201754628612768, -0.18840161629514585, -0.3024693517587498, -0.24059193377896793, 0.544032741035182, -0.03362151979717565, 0.45144412607650675, -0.3089056869698892, -0.3897585188774304, -0.05051286580809933, 0.14713872346755452, -0.2108582658832301, -0.5198967177980627, -0.5178694800927237, 0.35865134386590514, 0.4808114893162996, 0.789225641293567, -0.8508985514670566, 0.13823314491982797, -0.739675410243885, -0.15775135483951774, -0.03895169194849335, 0.7186066020769342, -0.023733552068291308, -0.758997755315213, 0.2223948964192961, -0.2839114983534181, 0.6804946967307284, -0.9674763927954904, -0.24194212418522842, 0.46271452163789983, 0.041958922514227615, 0.6252792076472264, -0.8933641411614821, -0.6742662534649226, -0.030415913481424806, -0.3391567178639272, 0.7195356190071083, -0.25280131191132016, 0.04233687946104148, -0.1985332024633999, -0.642597707975374, -0.949661131308059, 0.7819516063875755, -0.9706329660246342, 0.4555238713499983, -0.3244842304823301, -0.10291272128220097, -0.3769655198373316, -0.11879027249554075, 0.48139449312487526, 0.5746727162020351, 0.6597331554099224, -0.5433722008231474, -0.8801712954161591, -0.09455550791291722, -0.3882759692688944, 0.8413048179123557, -0.5396151720306099, 0.2828316370766222, 0.09268654419107736, 0.20502350469627562, -0.4702955826628239, -0.48419537096329623, -0.1102730678005841, -0.24597923928332088, 0.5136721091009624, 0.8299685814012263, -0.239278330859527, -0.6629570905291313, -0.03050376746546757, -0.6729060360229915, 0.00810948498408215, 0.24179729828206087, 0.3767891481914505, 0.3175324394130705, -0.3572109067127782, -0.5302967728173233, -0.40536179458770016, -0.07757636227034226, 0.24261486002815635, 0.3292762249342689, -0.30811789048060634, 0.942031456380366, -0.6250227690321042, 0.7684586895132757, -0.5105613490293044, -0.33327916905817534, -0.9777955468564694, 0.37754309424211363, -0.5407074000365668, -0.0010683200194339104, 0.12682835904731404, 0.7113855622033978, 0.712643245452947, 0.1901296076808634, 0.14783908949331792, -0.8714165782457974, -0.47671857734514256, 0.36775934955744427, 0.3582961894372194, -0.8700863873973963, -0.6746092108835755, -0.7084447361771433, -0.5305891682426807, -0.6248812381126891, 0.7354294206748293, 0.629945001274163, -0.0025439421573505427, -0.5779222263449557, -0.8943498684586193, -0.7615544705778559, 0.4418842048764158, 0.6536200085768438, 0.5293418915237016, -0.6034902758269745, 0.030595594267607273, -0.9580442963170717, 0.4095783080592521, -0.8926003716730673, -0.6886869672080218, -0.5944879224244224, 0.24757259399860732, 0.7422161750182508, -0.9628061599239992, -0.502338104507273, 0.32124493816668, -0.3114260044353736, -0.5628023031864373, -0.7506100650093699, 0.9999812273155888, 0.39476952267520726, -0.6227959502643867, 0.9104906866533928, -0.17608614578923798, -0.8676305330269367, -0.9281687708511557, 0.24655094057570226, 0.10848547772232653, 0.31619290107598585, -0.494516049049627, -0.14599989553223214, -0.9813611382060163, -0.6308652543161613, 0.7537366682149651, -0.2974748914211862, -0.4136736572637838, -0.737988693298258, 0.4119934124537472, -0.9973076772664051, 0.3463212026890037, 0.2138014946937281, -0.37515800737698024, -0.03813825432466622, 0.405608548039601, 0.6628569305805345, 0.9435593365235166, -0.771315619210891, -0.8287425590872795, 0.3605524369129378, -0.6985741961772283, -0.03919119865423237, 0.42585775584878016, 0.9666600817935671, 0.8417661534296137, 0.7721867122412354, 0.8185501545684908, 0.9688891694885744, 0.7662391850905297, 0.5149923238224556, 0.5009578794424276, -0.49328267071287124, -0.7106400056922211, -0.8768314388328209, 0.11500589472360923, 0.6087440746984127, 0.32842815481177556, 0.24867694511532856, 0.8008699577750804, 0.3840653196459862, -0.6868355420299008, 0.46028285143195213, 0.7900353184268263, 0.5059265749574271, -0.94716471382927, -0.6238302528103457, 0.7665077065696471, 0.8365834883101824, 0.2962040403287747, 0.6295416309721171, 0.44320583866024266, -0.9017841489316354, -0.4175652003936732, -0.9483704861320403, 0.8286183042553463, -0.3955990303576702, 0.41343889381614374, -0.9190084163324537, 0.8109623356116638, 0.3855792502480002, -0.30364789963646666, 0.36803654550631193, -0.9488793366707033, -0.4509695548516872, -0.36049978739833355, -0.7566263028085269, 0.8234539599714916, 0.8426288317292805, -0.6491428342822738, -0.7663998238707734, -0.0011577498153221288, -0.5438083202956341, 0.21223867198478086, 0.3546297782101451, 0.738673104663546, -0.6374814325963203, -0.12384537069058243, 0.6036127962495692, -0.5366898945981027, -0.06266630884165458, 0.08689973985244359, 0.8069723929476518, 0.4505642625155424, -0.931343550216345, -0.3202457727106611, -0.25838485659411314, 0.540971253204529, -0.46309960044704135, -0.8455421385036515, 0.07476159608239086, 0.5579067569583647, -0.04126405909713515, 0.05223770415962292, 0.18068478850746428, 0.405647406038502, -0.6442269440433328, -0.5786635947882115, -0.024806766349989573, -0.4561756268172401, -0.25928190778256655, 0.29703205987578896, 0.30443731610387026, 0.8292296987554253, -0.654939821503433, -0.7867403285021599, 0.8360225015098892, -0.8431427714843058, -0.731908510283005, -0.18488223883985833, 0.6325393607153451, 0.05488738070413923, 0.11503027279831057, 0.14001001172973204, 0.4246453134398991, -0.9593752476074036, -0.7216043186638024, -0.021092557175709903, -0.5928599422982588, -0.10036349587337123, -0.02323739211268494, -0.6832843968841664, -0.9199654173381491, -0.8642477281947554, 0.8180139072808179, 0.7907233245637093, 0.6847531029777587, -0.433056024706604, -0.1647290080417545, 0.6116204721063139, -0.20416105902103565, -0.37927773417532684, -0.08569007705226017, 0.9313922646794763, 0.1680333365168356, -0.9945074465901564, -0.009043767718076579, 0.35451795201235226, 0.9422478420299945, 0.3478680817206721, 0.033894064698831494, -0.9736225737382194, -0.6141513941463757, 0.3531366514798986, -0.9321299875731677, 0.3778329207733282, -0.8509485177312635, -0.8434385443308792, -0.3694102248264597, -0.8293701651832739, 0.7317017669814394, -0.809083920954127, -0.4358697506992748, 0.9729018168316499, 0.8971133267018863, 0.9733173382705784, -0.4616703346359312, 0.5087515899187847, 0.06359105410639643, -0.19465353558475473, 0.005757839122946251, 0.33091462839912356, 0.24968944323157238, 0.6292086697579711, -0.9997382093166731, -0.13727669731509695, 0.7868553870753983, -0.9860039639704172, -0.38509909892409366, 0.5590141040002339, 0.8879620503639158, -0.395978207010403, -0.7168591989169484, 0.41418042362862306, -0.9329118813002899, -0.03716052844489681, 0.08233834537451545, -0.7870595467068329, -0.04639423138898291, -0.4039927994434125, -0.7287079912585352, -0.9025159659953796, -0.9104327826596885, -0.08739405558699187, -0.9477703964745683, -0.40050500856368676, -0.10649202228582899, 0.21063871105872023, 0.8739322014787492, -0.07388379616738128, 0.9920144595314133, 0.33124377037056796, 0.5527925467238024, -0.6952348485582107, -0.16967677715511376, -0.275060423212669, -0.5091430192567243, 0.49293315781400815, 0.9611522101737986, 0.06612563877672328, -0.8712801744879586, -0.9126352862083515, -0.8309853725391523, 0.4777727806545973, 0.19627038491137205, 0.6382439072382609, -0.863715329492986, 0.3582254714393034, -0.3796693342964963, -0.7667423124784716, 0.7142329635667262, -0.2591100629572076, 0.29236190369118176, -0.5540874185777167, 0.1103798290924034, 0.4479196428347141, -0.23055770602932557, 0.3011309876038495, 0.1722852934633048, -0.31785599355951333, -0.38015947142214523, 0.1319736543507335, 0.6121712941249369, -0.32819812347084065, 0.4127123250760989, -0.3524116422088934, 0.9274737522546461, 0.9821303941180344, -0.4840363533670353, -0.7162562684498142, -0.386960335753016, -0.34919030821828345, 0.19982835312017055, 0.4083646101945224, 0.9848152466003999, -0.025347598856117814, -0.14376036233476386, 0.6881671014985931, -0.22436000125873967, 0.8308215248642383, 0.6434992268925035, 0.8929665097696429, -0.36183729379850904, 0.4856384826772584, -0.9991891373318273, -0.37208243100994554, -0.8427986685637934, -0.25171732149643233, 0.3743410696815763, 0.843011595205116, 0.9508180993750239, -0.38241482122758685, -0.4979848697535545, -0.8464707210136464, -0.5383773768802176, -0.9200109449454816, 0.3998942860990735, 0.1746425129153324, 0.9145507930820356, -0.9153270484027567, -0.714857771832395, 0.6870446342806231, 0.010971505953938587, 0.2807704341120143, 0.8648229477634561, 0.34984868614735576, -0.2514965354139729, -0.8378855896777828, -0.13638736270503093, -0.5391308373602639, -0.9590656647110205, -0.24688828735254842, 0.030049795497392395, -0.30049472757067797, -0.4635916370145261, 0.1522498408168651, -0.9695680567773379, 0.6735919980522744, 0.8472433307587048, 0.6939556513282221, 0.9655817234685438, 0.7324000964357515, -0.9486091020466045, -0.45306287601228457, 0.42468475030695974, 0.9471428808575824, 0.026887637547956977, 0.24702728704017263, -0.8403275596089463, 0.2572847626572925, -0.9588102847630264, 0.8845185886818274, 0.5112533801273591, -0.0003526048738131671, -0.7170211581545489, -0.23497342254346099, -0.7580982161676115, -0.17556956570548876, 0.5755017257659298, -0.2872117647411707, -0.29545648337319497, 0.45376431750014024, -0.03436696623840074, -0.522058689310406, -0.5936854670647158, -0.8518169555403163, -0.19110001513362684, 0.84599717510962, -0.16029111260394213, 0.08083189089401044, -0.14204231136682322, -0.2163894793590906, 0.7198407507091411, 0.1815091838125975, 0.9803447755458261, 0.9512016849158347, -0.07854568232728787, -0.638886475966213, -0.1318955082908273, -0.02548271113725953, 0.669851476591035, 0.20068182725364947, 0.3944271976490601, 0.39154101260559404, 0.8848620681101487, -0.3589242778841517, -0.7402424204087159, 0.8779481736578825, 0.10502392632069646, 0.11213822528920869, -0.5260364167237297, 0.9424374292566182, 0.2916560123065468, -0.8201735166552693, -0.40705968138699977, -0.05465398961618129, 0.3962193180206517, 0.11344387874989792, 0.036327765713952465, 0.7871311879491725, 0.4917396864287733, 0.4850497036236636, -0.9491852603701356, -0.8333613735116081, 0.16433342077894642, -0.44765957542047197, 0.8937810743513332, 0.02122908160404613, 0.834866525796548, -0.673610951492422, 0.07771961997375931, 0.7016396486038916, 0.43599195809425706, -0.8132413411863877, 0.11666268610761721, -0.9857345327314895, 0.57623686021015, -0.4623121157690966, -0.9256153413570902, 0.9670117696042111, 0.8098465670289063, 0.43340875870809215, 0.2661594596318493, 0.21518143336625806, -0.6585582781414205, 0.6024846127451537, 0.9499842479354961, -0.16688983979354544, -0.07764929201490234, -0.5030722618817207, -0.7480190881172333, -0.5600057606336057, 0.19473294701682464, 0.6620980588534056, -0.8434707369341092, -0.6965252281714704, 0.29201192120685926, -0.31706166251276446, 0.4176258743093135, 0.3613170203868197, -0.2888720870084076, -0.9797695719832715, 0.701748853583682, 0.18164123288377576, -0.3673184267165477, 0.853451950638771, -0.7406366083748395, 0.8441004488669088, -0.04966722965637005, -0.500804390763282, 0.7973909804233457, -0.7874809003259888, 0.7469735072773016, -0.0015601577933734845, 0.6017418084841943, -0.8684418347038643, 0.35820584520023924, 0.6076720049192994, 0.37151014908810365, -0.5881551691384643, 0.33782368978746047, -0.42666857626637134, -0.21112718996398128, 0.1442186049497347, 0.8791227017589711, 0.6102139167663354, 0.7324123489898164, -0.45737439348641806, 0.8327019511927438, -0.6098392718574228, -0.2160010289064418, -0.27874832902874025, -0.04087685078724723, 0.7572572230339774, 0.5960613358859412, 0.37650914707997996, 0.19239703294172883, 0.07881125416651114, 0.29363510795080905, 0.4597382589480401, 0.46712826953365716, -0.2183021756541479, 0.7813685253469125, -0.6369664690117813, -0.9569148489968957, -0.12368808246361307, -0.38057336435695843, 0.007925867388848706, 0.5110906966023283, -0.15509263621715452, -0.034703727194939527, -0.38581498378017853, 0.15867819781316062, -0.013077475727912846, -0.3983533978616238, -0.1241072515170547, -0.6140133830743968, -0.5725943496029406, 0.4947098678764974, -0.8860943736676861, -0.4106703765999793, -0.6274213396931723, -0.11995372048282138, -0.11946909699506181, -0.0682233084977899, -0.19582623638140229, -0.42103974804748945, 0.8206742993298406, -0.6272909906328594, 0.29855570919110974, 0.2870201614102632, -0.566965395448424, 0.592246717177128, -0.7645835085935717, -0.0010389585471823892, 0.6155393127662321, 0.9004966366869653, -0.40354896405709506, -0.6000780718575327, 0.9650378739776571, 0.46921179589955564, 0.8048470975020066, -0.3607250558057318, -0.7475069799611322, 0.4998461229314124, 0.15132515401587843, -0.7126235917408821, 0.700241197783612, -0.0032818752238510385, -0.16451929757939276, -0.3295631491354096, 0.5259436017562131, 0.4076811116622103, 0.7508866214896068, -0.5729697494953048, -0.9202114210188019, -0.2525621633746673, 0.4311146045780143, -0.058058158564393114, 0.578936571410887, 0.5424285731605141, -0.6877190870675904, -0.051797222109116525, -0.16791438600968656, -0.823748991775805, -0.12227599264580302, 0.2937172379641748, 0.04956313965586001, 0.6751341354820244, -0.36661501819904685, 0.16779811752550255, 0.2713897615219836, -0.05329613465736793, 0.43525615563202114, -0.30992988918622677, -0.015365501826986705, -0.968207821605864, -0.9042870597340944, -0.5687429370031691, 0.13967282908879275, 0.8236881113809296, 0.20804029575524607, -0.5069240608080532, -0.6154281995994788, -0.6280491399943937, -0.15461597349976564, 0.3052224944403428, 0.3162931668867188, -0.17814957682382593, -0.8269152917777638, 0.9867807676645342, 0.9231038935938112, -0.15759733727481162, 0.8512732582847622, -0.7803033499563774, 0.800547966515341, -0.8448963602984969, -0.4385856925501095, -0.477349694747778, 0.794040157843197, -0.8797719085050446, 0.760976775386571, 0.80195420072606, -0.9879619878882995, 0.7921905337865025, -0.822763124204638, 0.0369054268615332, 0.4973625177702343, -0.946128329452057, -0.2957277538361631, -0.6056526863206795, -0.2984959026309335, -0.48792040651919333, 0.9706868699446267, 0.8442476588588297, -0.9441173029954995, 0.7632350062301994, -0.039454840471977004, -0.6795875651976928, -0.006073280856235375, 0.28726682145919735, 0.8528034674812248, -0.7992888460850636, -0.44919853165850454, 0.9986316284975367, -0.3045496430635679, -0.7680160812007448, -0.4137819580710036, -0.3730875427884859, -0.18781630967146445, 0.6234570588359025, -0.3778345027413672, 0.14647710805669378]


