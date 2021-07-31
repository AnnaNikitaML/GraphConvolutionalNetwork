# GraphConvolutionalNetwork
This project is the result of compilation of our efforts on kaggle competition "Predicting Molecular Properties", https://www.kaggle.com/c/champs-scalar-coupling/overview. We uploaded our notebooks as they were at the time we worked on the competition. This is an attempt to store our efforts rather than to create a distributable implementation.

## Problem setting

The exact problem setting can be found on the competition web page. Here we will focus only on some interesting theory aspects and things that dictated our model choice.
The task is to determine scalar coupling constant between given pairs of atoms in the given molecules. 
What is the scalar coupling constant? To answer this question, let's first say a couple of words about nuclear magnetic resonance (NMR). Atom nuclei (those who have non-zero spin) have magnetic moment. Thus they can be aligned along applied magnetic field. 
Moreover, if we apply oscillating magnetic field perpendicular to that one, these nuclei start to oscillate in response. As with any forced oscillation, when the frequency of oscillator approaches the frequency of the external force, resonance occurs. 
What we are talking about here is not just an event of resonance, but its practical application, NMR spectroscopy. Different atomic nuclei have different spin and different mass, so they have different resonance frequency, given the same aligning external magnetic field. But there aren't so many elements with non-zero spin in the organic compounds: hydrogen, nitrogen, phosphorus, halogens, and some not-so-rare isotopes of other elelements, like carbon-13.<br><br>
So what't the catch? Nuclei in atoms and molecules are surrounded by cloud of electrons. These electrons orbiting the atoms partially shield the nuclei from external magnetic field. In molecules the electron clouds are spreaded across the atoms, and electron density, and, thus, magnetic shielding, varies with the exact location of the atom in the molecule. Different shielding leads to different effective field experienced by nuclei, affecting the resonance frequency. This is known as chemical shift.
Different nuclei of the same element resonate on different frequencies depending on their position in a molecule, thus, observing this certain frequencies one can deduce something on the molecular structure. Here we deal with proton NMR, and different protons in the molecule will have different chemical shift. But that's not all. Unlike someone could expect from classical physics, nuclei can align with or against applied external field (low and high enerhy states). One nucleus therefore can distort the magnetic field in its vicinity in either of two directions, affecting neighbouring nuclei. That slightly affects resonance frequency either slightly increasing or decreasing it. This cause signal split. The magnitude of this split, caused by one nucleus on another nucleus is known as coupling constant. While magnetic field created by one nuclei affects another, it also affects electrons and they affect another nuclei as well. This effect is known as indirect spin-spin coupling, or J-coupling, and this is what's this competition is about.

## Data description

For every molecule we have its structure (coordinates of atoms) and some additional data. This data can be specific to each atom (mulliken charges, magnetic shielding tensor), or molecule as a whole (dipole_moment, potential_energy). Training data also has scalar coupling constant, which we need to predict. Moreover, for training data we have four componets that contribute to this scalar coupling constant. We don't need to predict them, but can use this as an additional information.
The four componets that contribute to spin-spin coupling are: fermi contact (FC), paramagnetic spin-orbit (PSO), spin-dipole(SD), diamagnetic spin-orbit (DSO) contributions. For both training and test datasets we have pairs of atoms for which we need to determine the coupling constant, along with the coupling type (for example 2JHC, where 2 refers to the distance in molecular bonds between the two atoms, H and C refer to hydrogen and carbon respectively). One of the atoms is always hydrogen

## Brief description of the experiments done in the notebooks

- BondLengthes2507.ipynb

In this notebook we plot histograms of distance between the atoms. Based on these histograms we try to understand if we have bond between given pair of atoms and what type of the bond it is. Sometimes we see where separation is clear, sometimes it's not so straighforward. To support these observations we used tabular data for the typical bond lengths of different pairs of atoms. We draw 3D version of the molecules hightlighting the bonds found.

- GenFeatures0408.ipynb

Based on a set of thresholds form BondLengthes we try to determine multiplicity of the bond. In some cases we see clear separation on the histogram and can determine the bond for sure. That for example occurs for CN triple bond. In other cases, there is no clear separation between two peaks, and we introduce some smooth function, approaching integer multiplicity towards the peaks and farther ends of the peaks, while taking some intermediate value inbetween. We also use angles between bonds to determine the hybridization of atoms. We check on some examples whether hybridization of an atom corresponds its bonds multiplicity, and tried to tweak those interpolating functions to get slightly better fit. We don't normalize the total multiplicity of bonds around the atom to be integer. For example a lot of given molecules are polarized (for example having NH3+ at one end and COO- at another) and due to resonance carboxylate ion's oxygen atoms have 1.5 bond multiplicity.

- nn2608-calc_full.ipynb

We load all the data and import our Neural net. The result of our Neural Network (MyNet2) is atomwise, however, we need to know the information for the bond (pair of atoms). We need to pool the information along the path referring to J-coupling between given two atoms. We use another poolNet to combine this information together

- nn2708-eval.ipynb

We repeat the same steps as we do during training but we use validation data (10% of original training data). We represent the results after training and validation on the scatter plot for each type of coupling in order to check visually how close is the actual scalar coupling constant to the predicted one and whether the errors are similar for training and validation

- draw.py

Helper function to plot 3d balls-and-sticks molecule model. It really helps to visualise some interesting cases to better understand some corner cases.

- BFS.py

Helper functions to run BFS in a graph to search path between given two atoms. This is important when we have 2J or 3J coupling type (i.e. when the atoms aren't bonded directly) as we want to know via which atoms the path goes. Sometimes for 3J coupling we can have diamond-shape structure (so there are two paths of length 3 between the atoms), so we find all shortest paths, not just one.

- GeneratePairwisePathes0408.ipynb

Here we take the graph that represents a molecule and we find a path between two given atoms applying BFS as described above. We search these paths for all pairs of atoms for which the coupling constant is given (train case) or required to be found (test case).

- find_shortcuts0308.ipynb

The goal of this notebook is to draw some molecules for which BFS yields more than one path to better understand what are the typical examples of such structures could be.

- MyNet2.py

Graph Convolutional Recurrent Neural Network, takes an Adjacency graph for a molecule and the features for atoms, returns some latent features for atomwise.
