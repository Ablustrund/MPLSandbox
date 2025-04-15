# Providing feedback signals in RL

We validated the effectiveness of MPLSandbox in providing compiler feedback by integrating it into the RLCF framework, which significantly enhanced the code generation capabilities of LLMs. This code serves as a demonstration of using MPLSandbox's compiler feedback to provide reward signals for the PPO algorithm.

## Useage


First, clone the MPLSandbox repository and install it using pip.
```bash
git clone git@github.com:Ablustrund/MPLSandbox.git
cd MPLSandbox
pip install .
```

Then, clone the mplsandbox_for_rl repository and install the requirements.

```bash
cd mplsandbox_for_rl
pip install -r requirements.txt
```

Finally, run the training script to train the PPO agent using MPLSandbox's compiler feedback.
```bash
bash train_ppo.sh
```


## Citation

This project is based on [MOSS-RLHF](https://openlmlab.github.io/MOSS-RLHF/). If you use MPLSandbox or mplsandbox_for_rl in your research, please cite the following paper:

```bibtex
@misc{dou2024MPLSandbox,
      title={Multi-Programming Language Sandbox for LLMs}, 
      author={Shihan Dou and Jiazheng Zhang and Jianxiang Zang and Yunbo Tao and Haoxiang Jia and Shichun Liu and Yuming Yang and Shenxi Wu and Shaoqing Zhang and Muling Wu and Changze Lv and Limao Xiong and Wenyu Zhan and Lin Zhang and Rongxiang Weng and Jingang Wang and Xunliang Cai and Yueming Wu and Ming Wen and Rui Zheng and Tao Ji and Yixin Cao and Tao Gui and Xipeng Qiu and Qi Zhang and Xuanjing Huang},
      year={2024},
      eprint={2410.23074},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2410.23074}, 
}
```

```bibtex
@article{zheng2023secrets,
  title={Secrets of rlhf in large language models part i: Ppo},
  author={Zheng, Rui and Dou, Shihan and Gao, Songyang and Hua, Yuan and Shen, Wei and Wang, Binghai and Liu, Yan and Jin, Senjie and Liu, Qin and Zhou, Yuhao and others},
  journal={arXiv preprint arXiv:2307.04964},
  year={2023}
}
```

