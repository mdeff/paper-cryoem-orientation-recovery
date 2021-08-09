# Reviews at NeurIPS'21

Jelena: R2 + relevant todos
Mdeff : R4 + relevant todos

TO-DOs?
Write to AC that we find it unfair if the rating was lowered because of that? (Least comments and lower rate):  

That reviewer attributed the lowest score to our submission, which we find would be unfair if the reason for this low score is due to the reviewer's vision that this paper does not fall within scope of NeurIPS publications. As we argue in our rebuttal to Reviewer 1.
The other comments / feedback / criticism is (however) good / on point / justified.

Replace "General Rebuttal" with the appropriate name. 

## 2021-08-06 General rebuttal (reviewer-specific rebuttals below)

As the reviewers have rightly pointed out (and as we discuss in Section 4), the applicability of the proposed method to real practical situations is still conditioned on demonstrating its accuracy on "unseen" proteins (transfer learning). Once this is achieved, an extensive comparison with the most commonly-established pipelines in the field (cryoSPARC, Relion, etc.) is definitely required.

While it would have been ideal to deliver a new angle-refinement software for cryo-EM that is fully deployable in practice and competitive with the state-of-the-art, the task is a notoriously-challenging one: Cryo-EM measurements are some of (if not the most) noisiest data in biomedical imaging, and the global algorithmic process equates to a very-high-dimensional non-convex optimization problem. As a consequence, most of the current well-known cryo-EM processing packages are themselves the result of years of iterative refinement (no pun intended).

At the present, we have focused on proposing a new paradigm for estimating the orientations in cryo-EM, and have provided a first demonstration of the feasibility of this method in an (admittedly) simplified cryo-EM setting. We see the extension of the applicability of the method to a wider range of data and the comparison to other existing packages as a natural follow-up of this work, which we hope will soon be addressed in a separate contribution; we provide some pointers for this in Section 4.

### Working notes

We have preliminary results about transfer to unseen proteins. Could include, but decided against because they were not as robust as the other experiments. Would you see it as an improvement if we included them? In the Appendix?
TODO baselines / benchmark to SOTA: either done, either why not.
No experimental or real data because there wouldn't be true orientations to compare to. Such a comparison would require an entirely different evaluation pipeline, which we see as a separate contribution left for future work.

## 2021-08-04 Preliminary reviews

* Reviewer RNam: Rating: 5 / Confidence: 5
* Reviewer Z4BJ: Rating: 6 / Confidence: 4
* Reviewer 9yDH: Rating: 6 / Confidence: 4
* Reviewer zuH5: Rating: 4 / Confidence: 4

Average Rating: 5.25 (Min: 4, Max: 6)
Average Confidence: 4.25 (Min: 4, Max: 5)

## 1 - Official Review of Paper4764 by Reviewer zuH5
22 Jul 2021
NeurIPS 2021 Conference Paper4764 Official ReviewReaders: Program Chairs, Paper4764 Senior Area Chairs, Paper4764 Area Chairs, Paper4764 Reviewers Submitted, Paper4764 Authors

Summary:

The authors tackle the problem of image orientation estimation in single particle cryo-electron microscopy. This imaging technique generates a dataset of ~10^{5-7} 2D projection images of a 3D protein with unknown pose (elements of SO(3)xR2). Once poses are inferred, the 3D structure may be reconstructed with standard tomographic projection techniques. To estimate the distances, the authors propose a Siamese convolutional network architecture, which takes as input a pair of images, and attempts to predict the distance between their orientations (parameterized as quaternions). Once this distance function is learned (from a training set of posed images on a single protein), gradient-based optimization is used to infer each image’s pose. The authors show results on two small, synthetic datasets of 5000 images.

Main Review:

I appreciate the detailed study performed by the authors, however I do not think this paper is suitable for publication at NeurIPS as it focuses on a particular application without introducing any technical novelty, and the task is somewhat contrived for the application domain.

The task of ab initio reconstruction for homogeneous proteins is well-studied/solved. The authors should benchmark against a state of the art algorithm.

Fundamentally, the method relies on a training set of previously posed images, so the task they are addressing is not a realistic setting. As the authors discuss, if this approach were extended to be trained across multiple datasets and could predict orientations for a new (real) dataset zero-shot, it would be very high impact; however I am not convinced from the current results that extending this approach is promising. The paper lacks theoretical grounding, which the authors admit, and I am not convinced that there are any generalization properties for functions on distances between projection images (to unseen proteins).

Limitations And Societal Impact: Yes
Needs Ethics Review: No
Time Spent Reviewing: 3
Rating: 4: Ok but not good enough - rejection
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Code Of Conduct: While performing my duties as a reviewer (including writing reviews and participating in discussions), I have and will continue to abide by the NeurIPS code of conduct.

### Rebuttal

Thank you for your time and thoughtful comments.

We do think the paper is suitable for publication at NeurIPS. The novelty of our work lies in the tackling of a specific and important problem in computational biology with an innovative combination of (indeed existing) ML techniques; we believe this could open the door to new developments in ML for cryo-EM. Regarding the "contrivance for the application domain": The call for papers states that "NeurIPS 2021 is an interdisciplinary conference that brings together researchers in machine learning, [...], **computational biology**, and other fields" and specifically lists "Applications (e.g., speech processing, **computational biology**, computer vision, NLP)" as a topic of interest. Cryo-EM is one of the leading problems in computational biology nowadays, and its complexity obviously requires solutions that are tailored to the application domain. Hence, we do believe that the our paper perfectly fits within the publication scope of NeurIPS. 

Regarding the request to further extend the method to unseen proteins and compare it to existing pipelines: We refer the reviewer to the General Rebuttal, where those questions are addressed.

## 2- Official Review of Paper4764 by Reviewer 9yDH
16 Jul 2021
NeurIPS 2021 Conference Paper4764 Official ReviewReaders: Program Chairs, Paper4764 Senior Area Chairs, Paper4764 Area Chairs, Paper4764 Reviewers Submitted, Paper4764 Authors

Summary:

This paper uses deep learning to recover the orientations of molecules measured in the context of single particle cyroEM. The proposed method first estimates the pairwise distances between the angles of molecules based on the their projections. This is accomplished by embedding the projections into a feature space that was trained such that the cosine similarity between two features is equal to the distance between the orientations of the two molecules. Next, the authors solved an optimization problem (3) to recover the orientations, up to global ambiguities, from the pairwise distances.

Main Review:

Originality
Recovering cryoEM orientations or structure with DL is not new, but the previous works I'm familiar with have mostly focused on solving the task with dimensionality reduction. I haven't seen a paper tackle the problem by learning to estimate the pairwise distances between angles.

Quality
The paper did a good job of validating each proposed contribution in isolation. Simulations results were overall thorough, though there was no real experimental data.

Clarity
Well-written

Significance
It's not evident the proposed method will work with real data and the method was trained on the same molecules whose orientations it was going to reconstruct.
It's also unclear if the proposed methods improvements will actually improve the final quality of cryoEM reconstructions. This tasks performed in this paper are essentially preprocessing steps that recover an initialization that is fed into another algorithm (e.g., expectation maximization and RELION) that would reconstruct the structure. It's unclear if the improvements in the init help the end result.

Related work
Recent work in DL for single particle cryoEM has attempted to recover the 3D structure without explicitly recovering the angles using adversarial networks [A]. It may be worth mentioning. [A] Gupta, Harshit, et al. "Cryogan: A new reconstruction paradigm for single-particle cryo-em via deep adversarial learning." BioRxiv (2020).

Questions
It's my understanding that some molecules will preferentially align in certain directions, so that the view angles in cyroEM are not uniformly distributed. This is a major issue for embedding-based methods. How would non-uniformly distributed angles effect the proposed method?

Limitations And Societal Impact: Authors are transparent about the limitations of current method. Single particle cryoEM is an important problem.
Ethical Concerns: None
Needs Ethics Review: No
Time Spent Reviewing: 2.5
Rating: 6: Marginally above the acceptance threshold
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Code Of Conduct: While performing my duties as a reviewer (including writing reviews and participating in discussions), I have and will continue to abide by the NeurIPS code of conduct.

### Rebuttal

Thank you for your time and thoughtful comments.

Regarding the request to further extend the method to unseen proteins and compare it to existing pipelines: We refer the reviewer to the General Rebuttal, where those questions are addressed.

On how an improvement of the initial angle estimation translates into an improvement of the reconstruction: It has been shown in various works [R1, R2] that the outcome of iterative refinement procedures in cryo-EM is predicated on the quality of the initial reconstruction, or, equivalently, on the initial estimation of the orientations.
[R1] Carlos Oscar Sanchez Sorzano et al., "Optimization problems in electron microscopy of single particles", Annals of Operations Research, vol. 148,no. 1, pp. 133–165, 2006
[R2] Richard Henderson et al., "Outcome of the first electron microscopy validation task force meeting", Structure, vol. 20, no. 2, pp. 205–214, 2012

We will add the suggested reference to [A] in the revised manuscript.

On how non-uniformly distributed angles would affect the proposed method: We tried it in Appendix B (with a uniform sampling of Euler angles, which is non-uniform on SO(3), see Figure 12) and performance was not affected (see Figure 13).

## 3- Official Review of Paper4764 by Reviewer Z4BJ
16 Jul 2021
NeurIPS 2021 Conference Paper4764 Official ReviewReaders: Program Chairs, Paper4764 Senior Area Chairs, Paper4764 Area Chairs, Paper4764 Reviewers Submitted, Paper4764 Authors

Summary:

In this paper, “Learning to recover orientations from projections in single-particle cryo-EM,” the authors present an approach to recovering unknown projection orientations based on learning to predict the distance between projection angles given observed projections. Given these predicted projection angle distances, it is possible to recover the orientations themselves by solving for the orientations with distances that best match the predicted distances. The distances themselves are predicted using a Siamese neural network that is trained on projection pairs with known orientation distances. The authors demonstrate this approach on two synthetic datasets.

Main Review:

Although the method has some clear limitations and is only demonstrated on synthetic datasets, this paper presents a compelling and well described method and proof-of-concept demonstration. The paper is well written, and the problem and method are well motivated with relevant background. Specific comments follow below.

1. The big question: how will this work in practice on real cryoEM datasets where orientations are not known? Even training this on real cryoEM datasets is potentially problematic because the orientations of the particles were estimated to produce the reconstruction. Will a network be able to generalize over particles from different structures?
2. What happens when the particles are not derived from single rigid structure (i.e., there is conformational heterogeneity)?
3. When minimizing (4) to solve for the orientations from the distances, how are the orientations initialized? Similarly to t-SNE and other distance-based embedding methods, it seems like the objective is non-convex and so orientations may be mis-estimated due to poor initialization. How quickly does the minimization converge?
4. It would be useful to see a comparison with orientations estimated using a common lines approach (e.g., http://spr.math.princeton.edu/orientation). It would also be useful to see the orientations estimated by an iterative reconstruction procedure as baselines to better understand how well the orientations in these datasets can be estimated by other approaches. Overall, I think there are some nice ideas here that are well motivated and described. The main weaknesses are the question about how this will apply to real cryoEM data and the lack of strong baselines for orientation estimation.

Limitations And Societal Impact: Limitations and societal impact are sufficiently addressed.
Ethical Concerns: None.
Needs Ethics Review: No
Time Spent Reviewing: 3
Rating: 6: Marginally above the acceptance threshold
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Code Of Conduct: While performing my duties as a reviewer (including writing reviews and participating in discussions), I have and will continue to abide by the NeurIPS code of conduct.

### Rebuttal

Thank you for your time and thoughtful comments.

We have answered to your questions: 
1. The extension of the method to handle real cryo-EM datasets with unknown orientations is discussed in Section 4: "The success [of the method] as a faithful estimator eventually relies on our capacity to generate a synthetic training dataset whose data distribution is diverse enough to cover that of unseen projection datasets. Such realistic cryo-EM projections could be generated by relying on a more expressive formulation of the cryo-EM physics and taking advantage of the thousands of atomic models available in the PDB." We also refer the reviewer to the General Rebuttal, where this question is addressed from a higher-level perspective. 
2. This is indeed a critical point, which would deserve to be further discussed in a revised version of this paper. A pragmatic solution at this stage would be to rely on the 2D classification methods of existing cryo-EM packages, which try to untangle the different configurations within a cryo-EM projection dataset by classifying the 2D projections based on certain similarity criteria. Once the various configurations have been "separated", our method could then be run independently on each data sub-set. 
3. The orientations are randomly initialized, drawn from a uniform distribution over Euler angles. The objective function is indeed non-convex, though we found it to almost always converge to the same solution (up to a global rotation). We believe that initialization is not much of a problem here, as the space in which the embedding is optimized is the "true space", i.e., the space of 3D rotations SO(3), while methods like t-SNE embed in an Euclidean space of low dimension (for the purpose of visualization), which might not be able to accommodate / represent the data/samples. The orientation recovery minimization (4) convergence plots are shown in Figure 8 with blue color. The convergence time is reported in Appendix C and is ~3.75 hours (though Figure 8 shows it reaches a plateau 1-2 hours earlier).
4. Here as well, we refer the reviewer to the General Rebuttal, where the question of the comparison with existing packages is discussed. 

## 4 - Official Review of Paper4764 by Reviewer RNam
16 Jul 2021 (modified: 16 Jul 2021)
NeurIPS 2021 Conference Paper4764 Official ReviewReaders: Program Chairs, Paper4764 Senior Area Chairs, Paper4764 Area Chairs, Paper4764 Reviewers Submitted, Paper4764 Authors

Summary:

This work presents a method for estimating the viewing directions of projection images in single-particle cryo-electron microscopy (cryo-EM). The method consists of two steps, first the distance between the viewing angles are estimated using a neural network, then these distance estimates are combined to yield a viewing angle estimate for each projection image. The method is trained and evaluated on synthetic projection images, where it is shown to perform well in producing reconstruction distance estimates are combined to yield a viewing angle estimate for each projection image. The method is trained and evaluated on synthetic projection images, where it is shown to perform well in producing reconstructions of moderate resolution.

Main Review:

This paper presents an interesting method for ab initio reconstruction in single-particle cryo-EM, a difficult problem that stands to benefit from data-driven approaches due to the high noise content of the data, which is otherwise quite structured. The proposed method is promising, but numerical validation in the present work is lacking, especially with respect to comparisons with existing methods or experimental data. Instead, the performance of the method is evaluated in a synthetic-data setting without comparison to other methods and with a network that is trained on the same molecule. It is therefore difficult to judge the wider applicability of the method, despite its inherent appeal. If this type of validation could be provided (even in partial form), I would recommend this paper be accepted as part of the proceedings, but otherwise I ask that it not be included. More detailed comments follow.

As stated above, the problematic component of the work concerns the numerical validation. The motivation and description of the method is quite sound, but could be better explained in parts. For example, the authors switch back and forth between Euler angles θ_i and quaternions q_i (they are simply introduced without explanation on line 63), which can make for confusing reading. The authors also state that the mapping from S³ to SO(3) is a double cover, but do not explain whether this is a problem or not for the problem setting. On lines 120–121, the authors also discuss size invariance, but it is not clear why this is necessary. When are projections expected to be of different sizes? This should not be the case in the numerical experiments discussed since the training data and testing data arise from the same protein structure. A similarly confusing statement is on line 127, where the authors claim that “a space of n_f = 4 dimensions does not have room for G_w to represent other factors of variation”. What does this mean? Why are we constrained to have n_f = 4?

Constructing viewing angle estimates from a set of distance estimates also deserves some closer discussion in Section 2.3. Indeed, constructing a global set of angle estimates from smaller local distance estimates can lead to inconsistencies, as pointed out by Zhao and Singer, 2014 (Section 2.1). Why do we expect that constructing small subsets of the embedding at a time will result in a globally consistent embedding?

For the evaluation method described in Section 2.4, it is a bit problematic to use a non-deterministic error measure since it induces its own “error” for each run. It would be better if this could be replaced with a deterministic error. One way to achieve this is to replace the L¹-type norm used on the rotations and instead use a Frobenius norm on the rotation matrices. In this case, the optimal rotational alignment between the sets can be calculated by an SVD. That being said, perhaps the authors have other reasons for choosing the error measure that make it more appropriate than the simpler Frobenius norm. If that is the case, it should be motivated in the text.

For the numerical validation, it is striking that only uniform distributions of viewing angles are tested. First, these are relatively uncommon in experimental data, with a preferred orientation being the norm. Second, while many algorithms tend to work well for uniform distributions, that is not necessarily the case with non-uniform distributions. It would therefore be a good test of the approach to evaluate its performance in this regime.

A big problem with the evaluation is the choice of datasets. The authors acknowledge this in the discussion, but all the same, it would be good to include the results of some preliminary experiments here. For example, the networks are trained on one of two molecules. What happens if one network is used to predict the distances for the other dataset? Does performance suffer greatly or is there a degree of robustness here. What if the network is trained on both datasets? The universality of the approach will be crucial when extending the method to experimental data, so it is important to know what can be expected here, even at a preliminary level. Other questions related to training mismatch concern noise levels. To what extent can we train on one noise level and test on another?

When it comes to the noise levels, it is hard to extract a useful context. For example, the authors give the noise variance of the data, but without knowing the variance of the images, it is not obvious if this is a high or low noise level. Later in the text, we see that σ² = 16 corresponds to an SNR of –12 dB (which presumably means an SNR of 10 ^ (-1.2), but we are not given the formula). This is a low SNR, to be sure, but not exceptionally low for experimental cryo-EM data. Judging from the images shown in Figure 11 (which presumably have σ² = 16), these are indeed of moderate SNR for cryo-EM.

Another piece of the validation section that is missing is comparison to other methods. In the introduction, a host of alternative methods for ab initio structure determination are listed, but none are applied to the synthetic data used to evaluate the proposed method. It would be simple enough to run the synthetic data through the ab initio pipeline of Cryosparc, Relion, or Aspire to provide a simple baseline for comparison. Right now, the viewing angle accuracies and reconstruction resolutions are hard to place since there is no basis for comparison. This is especially important since it would provide an indication of the benefit derived from a data-driven approach, where specific features in the images are used to estimate the distances. Due to the training–testing regime, it is expected that the proposed method would perform better in this setting (since it possesses a much stronger prior compared to the baseline methods), but this needs to be validated numerically.

Some minor comments:
– Constant references to figures outside of the main text makes for tedious reading. I would try to include these in the main text (perhaps by cutting other figures) or remove references entirely. If this cannot be resolved, I suggest labeling these figures so that it is clear that they refer to supplementary material (Figure S1, etc.).
– On line 100, it should be clarified what is meant by the “magnitude” of a rotation.
– On line 102, what does it mean the “design” the estimator, and why is it intricate or impossible? What do invariants have to do with this?
– On line 120, a CNN provides equivariance (covariance) not necessarily invariance to translation (unless a pooling layer is included).
– On line 143, “our loss function” is referred to but has yet to be introduced.
– On lines 172, 244, 248, 252, and 268, spaces are missing between the number and the unit (Å).
– On lines 180, 191, and the caption of Table 1, there is an extra space after the comma in the numbers larger than one thousand.
– In footnote 6, sampling the Euler angles from uniform distributions does not yield a uniform distribution on SO(3). The colatitude (θ₂) has to be reweighted.
– On line 244, “an” should be removed.
– On lines 228–229, what does it mean that “the observed overfitting indicates that more training data should further decrease the sensitivity of the SNN to noise”?
– On line 240, “robuster” should be “more robust”.
– On line 253, why does higher resolution of the ground truth imply better reconstruction resolution?
– Several article and journal titles are improperly capitalized in the bibliography, including for references [11], [12], [18], [20], [21], [28], [30], [31], [32], [40], [41], and [43].

Limitations And Societal Impact: The authors discuss the limitations of their model and their numerical validation at the end of the manuscript. When it comes to the validation, I think more can be done (as stated in the main part of the review). This being an application of data-driven methods to a problem in structural biology, I do not see any potential negative societal impacts that would emerge directly from this work. The authors are, however, helpful in providing an estimated GHG budget for training their model.

Ethical Concerns: No ethical issues are raised by the manuscript.
Needs Ethics Review: No
Time Spent Reviewing: 2
Rating: 5: Marginally below the acceptance threshold
Confidence: 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
Code Of Conduct: While performing my duties as a reviewer (including writing reviews and participating in discussions), I have and will continue to abide by the NeurIPS code of conduct.

### Rebuttal

Thank you for your time and thoughtful comments.

**PUT GENERAL COMMENT HERE**

Thanks for pointing out minor presentation issues.
We switch back and forth between Euler angles θ_i and quaternions q_i as they are equivalent representations of the same information.
We'll try to remedy? TODO: What can we do?

The fact that the mapping from S³ to SO(3) is a double cover isn't a problem. It was meant as an explanation for equation 2.
We agree the writing could have been better and updated line 99 to read "The absolute value $\left| \cdot \right|$ ensures that $d_q(q_i, q_j) = d_q(q_i, -q_j)$ as $q$ and $-q$ represent the same orientation because $\mathbb{S}^3 \rightarrow \SO(3)$ is a two-to-one mapping (a double cover)."

> When are projections expected to be of different sizes?

That is indeed not the case with the current experiments but is expected to happen with real data.

> A similarly confusing statement is on line 127, where the authors claim that “a space of n_f = 4 dimensions does not have room for G_w to represent other factors of variation”. What does this mean? Why are we constrained to have n_f = 4?

The goal of that paragraph is to motivate our two-step approach (distance estimation then orientation recovery) over a simpler one-step approach of predicting the orientations directly from the projections.
In that case, the number of features n_f would be the size of a quaternion, i.e., 4 scalars.
We verified in Appendix F that such a space was too small for a good embedding.
We hypothesize that this is because the space is too small to capture all factors of variation: only the orientation is a factor of interest, the others ...
The number of features n_f can be chosen freely. One could hope to directly embed in the space of orientations and avoid making a detour through distances before embedding.
We wrote it as motivation for our two-step method, instead of the single step way of letting q_i = G_w(p_i).
The problem is that an embedding space of 4 dimensions is too small to capture other factors of variations, by which we mean variations which should be abstracted like the protein type.

> Why do we expect that constructing small subsets of the embedding at a time will result in a globally consistent embedding?

TODO: read Zhao and Singer, 2014 (Section 2.1).

How is the error measure in 2.4 non-deterministic?
We chose the mean orientation recovery error as it is simple to interpret what an average error of say 1° means.
TODO: we agree that we could also use a Frobenius norm on the rotation matrices. In this case, the optimal rotational alignment between the sets can be calculated by an SVD.
TODO: either add this error or motivate our choice in the manuscript.

We did evaluate our method on non-uniformly distributed viewing angles in Appendix B. Performance was barely affected.

To what extent can we train on one noise level and test on another?
As we don't know (yet) how to build NNs that are invariant to (specified) noise (and PSF), we need to resort to the brute-force trick of data augmentation.
To generalize / best test on any noise level, the NN should be trained on a variety of noise models (and PSFs).
Unlike transfer between proteins,
We are confident the NN would be good at that, as it was able to abstract noise well in our experiments.

The formula used to calculate SNR in dB is: $\text{SNR}_{\text{dB}} = 10 \text{log}_{10}(\text{SNR}), \text{SNR} = \frac{P_S}{P_N}$, where $S$ is a noiseless image, and $N$ is a noisy image with variance $\sigma^2=16$. We calculate $P_{S} = \sum_{i=0}^{M} \sum_{j=0}^{M} (s_{i,j}^2)$ and $P_{N} = \sum_{i=0}^{M} \sum_{j=0}^{M} (s_{i,j} - p_{i,j})^2$ with $M$ being the projection image width or height.
We agree the current formulation can be confusing. We'll add the SNR formula and give the image variance at the start or SNR instead of noise variance.

We do agree that the level of noise currently used in our experiments, although already very significant, does not cover the most severe cases of degradation observed in cryo-EM datasets. That being said, we believe that testing our method on synthetic cryo-EM measurements with a SNR of $-12$dB is a legit first step in demonstrating its potential for challenging real situations. Moreover, we have indications that ... // QUESTION@MDEFF, JELENA: Do we have some insights on the robustness of our method at higher noise levels?? 
Figure 7b shows performance for σ² from 0 to 25, corresponding to SNR of 0 and x, i.e., SNR_{dB} of -inf and log10(x).

TODO baselines / benchmark to SOTA: either done, either why not.
We are not at the stage where our method can be usefully compared to existing pipelines because we haven't ... transfer learning.
We don't have experience with Cryosparc, Relion, and Aspire.
CryoSparc, as the most automated pipeline of all, might be an option / we'll try to do it, etc.
The other packages unfortunately require too much tuning and previous experience to properly use.

Thanks for your minor comments. We'll address them in the revised manuscript. We answer the questions below.
* line 100: the angle of the rotation
* line 102: d_p is a function that could in principle be designed by a human (e.g., the Euclidean distance d_p(p_i, p_j) = || p_i - p_j || shown in Appendix E) instead of learned from data. We want that function to be invariant to noise. The problem being that we don't know how to design such a function, so we resort to learn it from examples.
* footnote 6: Agreed. And that's the raison d'être of the footnote. Some experiments have been done with a uniform distribution over SO(3), others (§3.2 and §3.4) with a uniform distribution over Euler angles. We empirically verified that sampling uniformly or non-uniformly over SO(3) didn't make a difference in Appendix C.
* lines 228–229: The SNN is overfitting the data, a sign that it wasn't trained on enough data. More data will make it generalize better.
* line 253: Because the projections/images are more detailed, leading both to an easier estimation of their viewing angles, and a more detailed 3D reconstruction.
