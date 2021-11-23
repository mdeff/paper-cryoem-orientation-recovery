# Learning to recover orientations from projections in single-particle cryo-EM

[Jelena Banjac](https://jelenabanjac.com),
[Laurène Donati](https://people.epfl.ch/laurene.donati),
[Michaël Defferrard](https://deff.ch/).

> A major challenge in single-particle cryo-electron microscopy (cryo-EM) is that the orientations adopted by the 3D particles prior to imaging are unknown; yet, this knowledge is essential for high-resolution reconstruction.
> We present a method to recover these orientations directly from the acquired set of 2D projections.
> Our approach consists of two steps: (i) the estimation of distances between pairs of projections, and (ii) the recovery of the orientation of each projection from these distances.
> In step (i), pairwise distances are estimated by a Siamese neural network trained on synthetic cryo-EM projections from resolved bio-structures.
> In step (ii), orientations are recovered by minimizing the difference between the distances estimated from the projections and the distances induced by the recovered orientations.
> We evaluated the method on synthetic cryo-EM datasets.
> Current results demonstrate that orientations can be accurately recovered from projections that are shifted and corrupted with a high level of noise.
> The accuracy of the recovery depends on the accuracy of the distance estimator.
> While not yet deployed in a real experimental setup, the proposed method offers a novel learning-based take on orientation recovery in SPA.

```
@inproceedings{cryoem_orientation_recovery,
  title = {Learning to recover orientations from projections in single-particle cryo-EM},
  author = {Banjac, Jelena, Donati, Laur\`ene, and Defferrard, Micha\"el},
  year = {2021},
  archivePrefix={arXiv},
  eprint={2104.06237},
  url = {https://arxiv.org/abs/2104.06237},
}
```

## Resources

PDF available at [`arXiv:2104.06237`][arXiv], [`OpenReview:gwPPcc_M0lv`][OpenReview].

Related: [code], [website].

[arXiv]: https://arxiv.org/abs/2104.06237
[OpenReview]: https://openreview.net/forum?id=gwPPcc_M0lv
[code]: https://github.com/JelenaBanjac/protein-reconstruction
[website]: https://jelenabanjac.com/protein-reconstruction

## Compilation

Compile the latex source into a PDF with `make`.
Run `make clean` to remove temporary files and `make arxiv.zip` to prepare an archive to be uploaded on arXiv.

## Figures

All the figures are in the [`figures`](figures/) folder.
The code and data to reproduce them is found in the [code repository][code].

## Peer-review

The reviews, decision, and our answers are in [`reviews.md`](reviews.md) and on [OpenReview].

## History

* 2021-08-09: rebuttal to NeurIPS'21 reviews (git tag `neurips-rebuttal`)
* 2021-06-04: submitted to NeurIPS'21 (git tag `neurips-submitted`)
* 2021-04-13: uploaded on arXiv (git tag `arxiv`)

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
