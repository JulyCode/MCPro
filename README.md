# MCPro: A Procedural Method for Topologically Correct Isosurface Extraction based on Marching Cubes [1]

We present an innovative procedural algorithm designed for extracting isosurfaces from scalar
data within hexahedral grids. The geometry and the topological features of the intersection of a level set of
the trilinear interpolant with the faces and the interior of a reference unit cell are analyzed. Ambiguities are
solved without the help of lookup tables, generating a topologically correct triangulation of the level set within
the cell that is consistent across cell boundaries. The algorithm is based on constructing and triangulating a
polygonal halfedge data structure that includes contours and critical points of the trilinear interpolant. Our
algorithm is capable of handling some singular cases that were not solved by previous methods.

## Usage

The algorithm is contained in the file `mc.py`.

`main.py` is used to test specific configurations.

By running `paper_images.py` the figures from the paper can be reproduced.

For the validation of the algorithm based on Etiene et al. [2] the test files in "Marching Cubes cases" and "Randomly generated grids" have to be downloaded from [here](http://liscustodio.github.io/C_MC33/) and placed into `src/data/`.
The `verifier.py` file can then be run to check for topological correctness.

## References

[1]:  J. Stahl and R. Grosso, "MCPro: A Procedural Method for Topologically Correct Isosurface Extraction based on Marching Cubes"

[2]:  T. Etiene et al., "Topology Verification for Isosurface Extraction," in IEEE Transactions on Visualization and Computer Graphics, vol. 18, no. 6, pp. 952-965, June 2012, doi: 10.1109/TVCG.2011.109.
