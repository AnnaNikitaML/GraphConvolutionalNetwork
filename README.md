# GraphConvolutionalNetwork
This project is the result of compilation of our efforts on kaggle competition "Predicting Molecular Properties", https://www.kaggle.com/c/champs-scalar-coupling/overview.

## Problem setting

The exact problem setting can be found on the competition web page. Here we will focus only on some interesting theory aspects and things that dictated our model choice.
The task is to determine scalar coupling constant between given pairs of atoms in the given molecules. 
What is the scalar coupling constant? To answer this question, let's first say a couple of words about nuclear magnetic resonance (NMR). Atom nuclei (those who have non-zero spin) have magnetic moment. Thus they can be aligned along applied magnetic field. 
Moreover, if we apply oscillating magnetic field perpindicular to that one, these nuclei start to oscillate in response. As with any forced oscillation, when the frequency of oscillator approaches the frequency of the external force, resonance occurs. 
What we are talking about here is not just an event of resonance, but its practical application, NMR spectroscopy. Different atomic nuclei have different spin and different mass, so they have different resonance frequency, given the same aligning external magnetic field. But there aren't so many elements with non-zero spin in the organic compounds: $^1H$, $^{14}N$, $^{31}P$, halogens, and some not-so-rare isotopes of other elelements, like $^{13}C$.
