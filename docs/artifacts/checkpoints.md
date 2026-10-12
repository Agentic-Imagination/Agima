# Checkpoints

[Artifact overview](README.md) · [Download guide](../data-and-models.md)

Links point to fixed Hugging Face revisions. The JSON manifest lists the files to download, their sizes, and available content hashes.

These exports contain EMA weights without optimizer or scheduler state. Download the BAGEL base configuration, tokenizer, and VAE separately. Set the runner's planning format, K, image preprocessing, and decoder according to the model card and [evaluation guide](../evaluation.md#select-a-checkpoint-contract).

## Chunk policies

| Task | Artifact |
| --- | --- |
| Maze2D | [maze2d_easy_cot_chunk_k3_world_model](https://huggingface.co/novastar111/maze2d_easy_cot_chunk_k3_world_model/tree/231f25bfba80be61feaae0040ebced8c7c2acb50) |
| Maze2D | [maze2d_easy_cot_chunk_k5_world_model](https://huggingface.co/novastar111/maze2d_easy_cot_chunk_k5_world_model/tree/db30142e05bd2c206b2151fe096af54010127583) |
| Maze2D | [maze2d_easy_cot_chunk_k10_world_model](https://huggingface.co/novastar111/maze2d_easy_cot_chunk_k10_world_model/tree/3234c4474095d02fb26e74d19865b63d21321e54) |
| Maze2D | [maze2d_easy_cot_chunk_kinf_world_model](https://huggingface.co/novastar111/maze2d_easy_cot_chunk_kinf_world_model/tree/20e4c20e04863b7b32d8cd494f5326bdff5b6a28) |
| Maze2D | [maze2d_easy_noncot_chunk_k1_world_model](https://huggingface.co/novastar111/maze2d_easy_noncot_chunk_k1_world_model/tree/85db4b241c357cee5e72cb33114cbfa152e31c1e) |
| Maze2D | [maze2d_easy_noncot_chunk_k3_world_model](https://huggingface.co/novastar111/maze2d_easy_noncot_chunk_k3_world_model/tree/881fae8bb66c97ab7a231ca7fdb51cf499e0dd94) |
| Maze2D | [maze2d_easy_noncot_chunk_k5_world_model](https://huggingface.co/novastar111/maze2d_easy_noncot_chunk_k5_world_model/tree/b92b041eeda871a53bc6f3cd215c1bb253456ba7) |
| Maze2D | [maze2d_easy_noncot_chunk_k10_world_model](https://huggingface.co/novastar111/maze2d_easy_noncot_chunk_k10_world_model/tree/c63780f5680e46e20a1285044822860df0e27f63) |
| Maze2D | [maze2d_easy_noncot_chunk_kinf_world_model](https://huggingface.co/novastar111/maze2d_easy_noncot_chunk_kinf_world_model/tree/d9c948ab840b007926a4f04dc47c0784b98cf6bf) |
| Pacman | [pacman_braidthin_cot_chunk_k3_world_model](https://huggingface.co/novastar111/pacman_braidthin_cot_chunk_k3_world_model/tree/ed662efc1fc9c0e7f79955e4b2af1ae40f936bac) |
| Pacman | [pacman_braidthin_cot_chunk_k5_world_model](https://huggingface.co/novastar111/pacman_braidthin_cot_chunk_k5_world_model/tree/a639e1ba19eccd6708c160110e77b6445888463a) |
| Pacman | [pacman_braidthin_cot_chunk_k10_world_model](https://huggingface.co/novastar111/pacman_braidthin_cot_chunk_k10_world_model/tree/e092f34e94acecd71bad2546bcf4b4e8b9810190) |
| Pacman | [pacman_braidthin_cot_chunk_kinf_world_model](https://huggingface.co/novastar111/pacman_braidthin_cot_chunk_kinf_world_model/tree/12a0c9314cca1c27a229818ca293badc460772de) |
| Pacman | [pacman_braidthin_noncot_chunk_k1_world_model](https://huggingface.co/novastar111/pacman_braidthin_noncot_chunk_k1_world_model/tree/c3653e5a4752a184899acc58993de1a17025d4f1) |
| Pacman | [pacman_braidthin_noncot_chunk_k3_world_model](https://huggingface.co/novastar111/pacman_braidthin_noncot_chunk_k3_world_model/tree/3a8455287c7d5c3593c86b00bff0018a8b6a9c1c) |
| Pacman | [pacman_braidthin_noncot_chunk_k5_world_model](https://huggingface.co/novastar111/pacman_braidthin_noncot_chunk_k5_world_model/tree/1de20f9f7757c8fad2dbacf1c34ebb7c426220c6) |
| Pacman | [pacman_braidthin_noncot_chunk_k10_world_model](https://huggingface.co/novastar111/pacman_braidthin_noncot_chunk_k10_world_model/tree/e48c4cfe1627b0263d032b80c1876826d92eed96) |
| Pacman | [pacman_braidthin_noncot_chunk_kinf_world_model](https://huggingface.co/novastar111/pacman_braidthin_noncot_chunk_kinf_world_model/tree/1fe9408b7f9a7123a247f42d6c6cfffa5ab28ebe) |
| PushT | [pusht_cot_chunk_k3_world_model](https://huggingface.co/novastar111/pusht_cot_chunk_k3_world_model/tree/d1885dea929433aa8c97dc579376c6a34c604aeb) |
| PushT | [pusht_cot_chunk_k5_world_model](https://huggingface.co/novastar111/pusht_cot_chunk_k5_world_model/tree/391c75622b2b8e2354123aa147bd80e7bd99b386) |
| PushT | [pusht_cot_chunk_k10_world_model](https://huggingface.co/novastar111/pusht_cot_chunk_k10_world_model/tree/b70b5e8dea466f40567e9063a8e901ca20717a01) |
| PushT | [pusht_cot_chunk_kinf_world_model](https://huggingface.co/novastar111/pusht_cot_chunk_kinf_world_model/tree/07bdd17e15fc7ac8937de6694353eefa8d24bae9) |
| PushT | [pusht_noncot_chunk_k1_world_model](https://huggingface.co/novastar111/pusht_noncot_chunk_k1_world_model/tree/b0d715edf0a25c13600d7c86f56d7944fd7568b4) |
| PushT | [pusht_noncot_chunk_k3_world_model](https://huggingface.co/novastar111/pusht_noncot_chunk_k3_world_model/tree/99d9f7dbf7747ab06c9c6c775c49ee90fe08ab1f) |
| PushT | [pusht_noncot_chunk_k5_world_model](https://huggingface.co/novastar111/pusht_noncot_chunk_k5_world_model/tree/b6b82303542475a91d5188dda5c21f4526b8e72d) |
| PushT | [pusht_noncot_chunk_k10_world_model](https://huggingface.co/novastar111/pusht_noncot_chunk_k10_world_model/tree/3a537a1425165bfd887f291f421af4cd47223caf) |
| PushT | [pusht_singleshot](https://huggingface.co/novastar111/pusht_singleshot/tree/a19674153e5fbe1c5d2b816f6da5d9c9fef94497) |
| Sokoban | [sokoban_easy_cot_chunk_k3_world_model](https://huggingface.co/novastar111/sokoban_easy_cot_chunk_k3_world_model/tree/b8e525e706517c92f7ba914b4e5c03f9491a853d) |
| Sokoban | [sokoban_easy_cot_chunk_k5_world_model](https://huggingface.co/novastar111/sokoban_easy_cot_chunk_k5_world_model/tree/887196bd83c0f3ccd1ad99eb6d468148011de327) |
| Sokoban | [sokoban_easy_cot_chunk_k10_world_model](https://huggingface.co/novastar111/sokoban_easy_cot_chunk_k10_world_model/tree/2ce54d2672d40c4277077845f8a529c804b14cb0) |
| Sokoban | [sokoban_easy_cot_chunk_kinf_world_model](https://huggingface.co/novastar111/sokoban_easy_cot_chunk_kinf_world_model/tree/4750d706f82ab88c8c8e22cf66cad4684e99cdc0) |
| Sokoban | [sokoban_easy_noncot_chunk_k1_world_model](https://huggingface.co/novastar111/sokoban_easy_noncot_chunk_k1_world_model/tree/49d95e9df3ed935d5ded0370fdab3010ea0d1794) |
| Sokoban | [sokoban_easy_noncot_chunk_k3_world_model](https://huggingface.co/novastar111/sokoban_easy_noncot_chunk_k3_world_model/tree/a30942dec40a4bda466d31f3a4701e44c46f7d58) |
| Sokoban | [sokoban_easy_noncot_chunk_k5_world_model](https://huggingface.co/novastar111/sokoban_easy_noncot_chunk_k5_world_model/tree/b14f57ff51af94b5215b8c3e227687c3f5f40a25) |
| Sokoban | [sokoban_easy_noncot_chunk_k10_world_model](https://huggingface.co/novastar111/sokoban_easy_noncot_chunk_k10_world_model/tree/be4b7e7c12b91b0a2bf5c41df1b08a5d0f980097) |
| Sokoban | [sokoban_easy_noncot_chunk_kinf_world_model](https://huggingface.co/novastar111/sokoban_easy_noncot_chunk_kinf_world_model/tree/13893d9643272dd6aa4221c6e3788daa9a679e82) |

## Planning

| Task | Artifact |
| --- | --- |
| Pacman | [pacman_braidthin_always_forward_kinf_world_model](https://huggingface.co/novastar111/pacman_braidthin_always_forward_kinf_world_model/tree/e278978ea497c6d545113c3edee6ea7bc493decf) |
| Pacman | [pacman_braidthin_forward_v2_cutoff50_kinf_world_model](https://huggingface.co/JiaxinGe/pacman_braidthin_forward_v2_cutoff50_kinf_world_model/tree/35dd4bfa48a8fbb6e6693fb738b048aba5d55658) |
| Pacman | [pacman_braidthin_forward_v3_mixed_kinf_world_model](https://huggingface.co/JiaxinGe/pacman_braidthin_forward_v3_mixed_kinf_world_model/tree/ab1884d3285fe91da7db0deb2ef717a28d539bd5) |
| Pacman | [pacman_braidthin_forward_v4_shorttrap_only_kinf_world_model](https://huggingface.co/JiaxinGe/pacman_braidthin_forward_v4_shorttrap_only_kinf_world_model/tree/5cf58abe00e807c28708fb6dec3de63da589b0b3) |
| Sokoban | [sokoban_adaptive_mixed_branch_sft3k](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_sft3k/tree/5d357a98c7967903b59e3b2c91fa12353625ee98) |
| Sokoban | [sokoban_adaptive_mixed_branch_sft5k](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_sft5k/tree/4f3025b0617b983ceda851bcb3b5a7cb7d905533) |
| Sokoban | [sokoban_adaptive_mixed_forward_sft3k](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_sft3k/tree/058af542ffecb3d3ad681b39d3e1b9185cf53f08) |
| Sokoban | [sokoban_adaptive_mixed_forward_sft5k](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_sft5k/tree/47b78d5552801650263684b585dde2be99fdac02) |
| Sokoban | [sokoban_adaptive_vanilla_sft3k](https://huggingface.co/novastar111/sokoban_adaptive_vanilla_sft3k/tree/ec25fc143e2152b9f09ebd350bcd77b7215e10cd) |
| Sokoban | [sokoban_easy_cot_chunk_kinf_branch](https://huggingface.co/novastar111/sokoban_easy_cot_chunk_kinf_branch/tree/67b8eea35192bb628e35f73762283cc1f5a4e91a) |
| Sokoban | [sokoban_easy_cot_chunk_kinf_compare2](https://huggingface.co/novastar111/sokoban_easy_cot_chunk_kinf_compare2/tree/e2e8fb48d30ceddaa15199abe5ec56ae85edb23c) |

## Reinforcement learning

Each entry identifies the initialization, training setting, and saved RL update in its repository name. These exports require the matching research planning runner; they are not interchangeable with finite-K launcher recipes.

| Task | Artifact |
| --- | --- |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step10](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step10/tree/2fa94e56e13f5ea6242525aabbf9714f0b4226ee) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step20](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step20/tree/11564a8a5e7d2d39df9b2e23c27c8d3a8b471c61) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step30](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step30/tree/038673a0db32195c81e796e9df84f40ea87d68cf) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step40](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step40/tree/9f938339a070e14bf6608e1feccb5c37ca46d465) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step50](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step50/tree/1dddac77ec9b9e90d1b7839d19d7102428e1e90f) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step60](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step60/tree/b698ab06930a78b546488c8de3e06b5f38045b80) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step70](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step70/tree/1964f7e57dd09c6968e8706306aa50746fcd6db6) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step80](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step80/tree/7e199e2754e52e8dee475b4fa8a2df5bff44e90e) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step90](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step90/tree/3d8d1615d88eeadfa81900a77924395688e51a33) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step100](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step100/tree/221b192e6378b08ac42222c59f9de11a0d742b66) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step110](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step110/tree/c7cd4a755470395ad8053f7646f780f46b1ec53e) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step120](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step120/tree/0edfadcb566e2bcf3af160ef7dac840ccdf04b43) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step130](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step130/tree/97d6d0326cc051f86dfaa0cbf6d5b291007ecabe) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step140](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step140/tree/03678a97b36e1ed32213c9240dc03a271aae84f4) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step150](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step150/tree/10918ff6f509d48c50b976c3d2be3fbf3bd1de75) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step160](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step160/tree/3197068420628d58f996e686caaeb3b2363bd7db) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step170](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step170/tree/35576f3025c5a72cf4dbcdddf2f28b20ad2fa8fd) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step180](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step180/tree/9eeeb1651fab81ec57b11f487a251f62fb809e2c) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step190](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step190/tree/dcdbd9c490308ce9d517ba7dedcdcba14c59aeab) |
| Sokoban | [sokoban_adaptive_mixed_branch_rl_step200](https://huggingface.co/novastar111/sokoban_adaptive_mixed_branch_rl_step200/tree/c7edfa84f99424e8d32630ff7f5e46cecd84ad17) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step10](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step10/tree/259067105c849dbba574c1b2eb19988aeff5ebcb) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step20](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step20/tree/f66c1c21872b3ceebcccec580d422a04ec5afffd) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step30](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step30/tree/e491c675d30e01716e5cc6779acda8adc2e2fa7f) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step40](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step40/tree/a2e475dfcf2de4357b22437ad634798dadb3bb2b) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step50](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step50/tree/c936cc5fdaa375c5dcc9f0557fb9f85c8c7217e2) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step60](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step60/tree/060bedfc29b7ae87fafa9550f7d9cd3cbd022d1a) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step70](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step70/tree/68ecd8cb4acbfa401fc3542367da44f34d3a62ff) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step80](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step80/tree/1e70953584dfd81343294be1508078a641336be0) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step90](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step90/tree/4c81bde01eea8f67e125cb2de17c42c348357114) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step100](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step100/tree/e6a364167779a649dd4d29d7cf41f604e4894d0c) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step110](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step110/tree/428fe65592bc8f655dca97524929ed04357271bb) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step120](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step120/tree/b0bf3858f14f35548413bb658a9f56e17a0f8691) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step130](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step130/tree/87b5c1435bdf476282d5b91d344bb8639f8e9f10) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step140](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step140/tree/5a84c7e0fb581c93fce170e1753b76f670b96da7) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step150](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step150/tree/f29d6f4c58af4f328ff02f13cd84c19cb4b8c3eb) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step160](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step160/tree/be98db3f71ebc5f1600aefb3e71ab73768586fe0) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step170](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step170/tree/d66c370b4a72d1743af522d100c408ae840928b8) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step180](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step180/tree/950354925f53a790777a58ff114bc01b10cdc055) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step190](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step190/tree/ef20d161d55b272f58bd25edf222c9729a216c66) |
| Sokoban | [sokoban_adaptive_mixed_forward_rl_step200](https://huggingface.co/novastar111/sokoban_adaptive_mixed_forward_rl_step200/tree/493ba60f1945de51a97a8d6a4e01b083cdc7922a) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step10](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step10/tree/da245a873b0dabe0ab0892e2a82184d9eca2ad73) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step20](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step20/tree/9f8c4bb5ab3916085482b3e3f87d61414531576f) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step30](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step30/tree/0463ddda007278a6a8122ec2a19f4d52ff185662) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step40](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step40/tree/d236aa8e22535d73df1bc8fdc4ab0ed1dbc70c3f) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step50](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step50/tree/71bbf95dd3caaf81c50b1b03b46268292d9c7a99) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step60](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step60/tree/9bba708e33b30b30f5c5be00cf8a896eea81f050) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step70](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step70/tree/febd57b5b9148c81288a5bb03c7372afab4abec6) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step80](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step80/tree/0cd422f8f254f2c28df2d80eec85bced7ed4db31) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step90](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step90/tree/137be3e2a2b91991e54d98adac5e9fa6cf2d9c77) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step100](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step100/tree/03c385d7dd2f50ec2b05a5305a1d3c0aa314a74c) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step110](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step110/tree/07f18de9c066e3a228350955acca5e33fd1b3f5f) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step120](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step120/tree/a8506bb9b1e5ba2f19221634087c0bd74bee2c67) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step130](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step130/tree/ee99836ca572765a6288031d91031a50fbef2392) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step140](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step140/tree/89c9a5280f627623314f4f6bc7a7feaae6eeed4e) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step150](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step150/tree/a66eab4ced59a40c1446c95ea89e11712a921752) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step170](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step170/tree/d86a9ba994ebe0fed97892f02a39dde0a0ccee8c) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step180](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step180/tree/fa67d5c01852bd919ca89ca740ab5aee59c6934b) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step190](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step190/tree/cc68ebbf7025102b8e4e27af9b2fe2a08ae593f1) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl2box_step200](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl2box_step200/tree/de451f642c05188fce799f3ceb00531073e7b8bc) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step10](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step10/tree/3dae038cd11e1870ea9a385405a7a6b835d70904) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step20](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step20/tree/5d6ece679608e7d1c19126bc835ea15465d40992) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step30](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step30/tree/027d4b654e933bf4c73507d166049b4307acc520) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step40](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step40/tree/8d51d021741d9b42da9f7251a1137fe346478131) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step50](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step50/tree/41ad267064bc4dc6115d6e624c6ba95e44e5dd6d) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step60](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step60/tree/d64364e7cd129d46f54ebd286eaa610bd143de0e) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step70](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step70/tree/50a286cdd21876587b33eb781979d682f6b4a5ad) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step80](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step80/tree/71790724eac59ed78a50af863900b8303874ab01) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step90](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step90/tree/c900ddde8806085f6538a3c5712683332809fb3c) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step100](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step100/tree/8fbdf014269559a30e7471c2c227ee6ced90d51e) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step110](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step110/tree/f10c522ff318d2b53bbeca35bb3a77298956c9d5) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step120](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step120/tree/17f6d0d9442073961fffbfb7601ad7651f0aac2b) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step130](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step130/tree/434b62a3a5ba26554e596758286588c0f902585f) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step140](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step140/tree/c4f6ac76bde9652cec75e44ec7c115872fb89eb0) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step150](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step150/tree/3a6c48a2f3e8b612320a48a9124401a93a43d867) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step160](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step160/tree/6b772386a33634e4bb88194ac264f17cfb16c91c) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step170](https://huggingface.co/novastar111/sokoban_adaptive_vanilla3k_rl3box_step170/tree/eb7406ee48dabfc5c73fa15e953c44d5f8ee90df) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step180](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl3box_step180/tree/af919d3519f070432b217bd671383885c83b0df9) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step190](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl3box_step190/tree/28b04d98dcdd179c76894fd88dec2bc0572b34d7) |
| Sokoban | [sokoban_adaptive_vanilla3k_rl3box_step200](https://huggingface.co/JiaxinGe/sokoban_adaptive_vanilla3k_rl3box_step200/tree/6127132757b784739ecc9bf5aeaf9c27d6efd572) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step10](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step10/tree/e91810a6f0c8ba08a666beb8e890c3ca08518ce6) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step20](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step20/tree/b11dea50a2278162b7ee9c995b0cf00f2c40672a) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step30](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step30/tree/da55edaa90e38c05d5a8ae345879df1151d26467) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step40](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step40/tree/68f86be2011cd296911105438b898287cb43d3ff) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step50](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step50/tree/bff2f7e83d4891cd77f37f1eef09588dd7a24334) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step60](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step60/tree/a8f481b201a54f8f93bbd5dafa3f25c001edbde9) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step70](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step70/tree/eb39067257ccab7ffdba155084ff648c13a57217) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step80](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step80/tree/05509cf62ad5465ef2228c9a5ede250267e768a7) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step90](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step90/tree/3e141ca0a89bec50f18fb7fed28557dfdb611036) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step100](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step100/tree/5d9ee3621d4a7c498748000b6b1c14daca16c885) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step110](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step110/tree/147f2829a04abdd031ece6de861ccb330199a3f1) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step120](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step120/tree/b850625aa679ceff1bbe348d5629750ecf1b9e79) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step130](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step130/tree/753969e04324662b4a6c05a69de0bc0a51864008) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step140](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step140/tree/c6ecf298f29634775bd362ff27a8d10aa3230082) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step150](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step150/tree/55d6f02aefe0edc9bcf65b0e50dc888e10b45e12) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step160](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step160/tree/1f666a8c4d322ef59ef7af46382fdaa3eb082631) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step170](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step170/tree/f9bb86e7f08a0bc13fa7fda0e970e5af3e476d39) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step180](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step180/tree/210a223ac4b1ec6f13c7afbe42fce489eed5f23d) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step190](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step190/tree/b13d70c3d027fb44819fe65b6ee7c9a01c5a9f90) |
| Sokoban | [sokoban_adaptive_vanilla5k_rl_step200](https://huggingface.co/novastar111/sokoban_adaptive_vanilla5k_rl_step200/tree/884c6b1f778a171f5a55761eff0c6d192c0d5cb1) |

## SFT controls

| Task | Artifact |
| --- | --- |
| Sokoban | [sokoban_canonical3k_originaldata_continuation_compute_match_rl50](https://huggingface.co/JiaxinGe/sokoban_canonical3k_originaldata_continuation_compute_match_rl50/tree/043ed50541aef1570a76524a0eae837c06aac16a) |
| Sokoban | [sokoban_canonical3k_originaldata_continuation_compute_match_rl100](https://huggingface.co/JiaxinGe/sokoban_canonical3k_originaldata_continuation_compute_match_rl100/tree/b797880c6e56e401e5f2e54e9300ea55f47f95f5) |
| Sokoban | [sokoban_canonical3k_originaldata_continuation_compute_match_rl150](https://huggingface.co/JiaxinGe/sokoban_canonical3k_originaldata_continuation_compute_match_rl150/tree/89221cc982fed2ba9d453381f1dc4ea3a7075991) |
| Sokoban | [sokoban_canonical3k_originaldata_continuation_compute_match_rl200](https://huggingface.co/JiaxinGe/sokoban_canonical3k_originaldata_continuation_compute_match_rl200/tree/c5434f456cb0aebdc509e3654c66f2ed30770ae5) |
