<p align="center">
  <a href="https://agima.net/"><img src="docs/assets/agima.svg" width="80" height="80" alt="Agima logo"></a>
</p>

<h1 align="center">Training an Imaginative Agent</h1>

<p align="center">
  <a href="https://gejiaxin.org/">Jiaxin Ge</a><sup>1,*</sup> &nbsp;
  <a href="https://www.junyi42.com/">Junyi Zhang</a><sup>1,*</sup> &nbsp;
  <a href="https://agima.net/">Tianning Chai</a><sup>1,*</sup> &nbsp;
  <a href="https://github.com/lukehanjun">Hanjun Yoo</a><sup>1</sup><br>
  <a href="https://agima.net/">Chenyang Wang</a><sup>2</sup> &nbsp;
  <a href="https://xiuyuli.com/">Xiuyu Li</a><sup>2</sup> &nbsp;
  <a href="https://weijiashi.notion.site/">Weijia Shi</a><sup>3</sup> &nbsp;
  <a href="https://xudongfrankwang.github.io/">XuDong Wang</a><sup>1,4</sup><br>
  <a href="https://www.sewonmin.com/">Sewon Min</a><sup>1</sup> &nbsp;
  <a href="https://www.cs.washington.edu/people/faculty/lsz/">Luke Zettlemoyer</a><sup>3</sup> &nbsp;
  <a href="https://people.eecs.berkeley.edu/~trevor/">Trevor Darrell</a><sup>1,†</sup> &nbsp;
  <a href="https://havenfeng.github.io/">Haiwen Feng</a><sup>1,2,†</sup>
</p>

<p align="center">
  <sup>1</sup> University of California, Berkeley &nbsp; <sup>2</sup> Impossible Research<br>
  <sup>3</sup> University of Washington &nbsp; <sup>4</sup> Duke University<br>
  <sup>*</sup> Equal contribution &nbsp; <sup>†</sup> Equal advising
</p>

<p align="center">
  <a href="https://agima.net/">Project page</a> · <a href="https://agima.net/#results">Demos</a>
</p>

https://github.com/user-attachments/assets/7a7dd69e-b6bb-4a99-b013-f2aad7504183

Agima learns to plan through visual imagination: it generates actions interleaved with predicted future observations, then executes them in the environment. We study Sokoban, PushT, Maze2D, and Pacman using [BAGEL](https://github.com/ByteDance-Seed/Bagel).

![Agima method](docs/assets/method.webp)

## Resources

| Resource | Contents |
| --- | --- |
| [Models and data](docs/checkpoints.md) | Checkpoints and datasets reported in the paper. |
| [Installation, training, and evaluation](docs/usage.md) | Setup and commands in one page. |
| [Image ablations](experiments/image_ablation/README.md) | Original, Blank, Stale, and Oracle GT experiments. |

## Acknowledgments and license

Agima builds on [BAGEL](https://github.com/ByteDance-Seed/Bagel) and Gymnasium. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md) for source terms, including the CC BY-NC 4.0 component. Models and datasets retain the licenses in their respective cards.
