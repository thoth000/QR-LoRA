# QR-LoRA デモスクリプト詳細説明

このドキュメントでは、QR-LoRAプロジェクトの各デモスクリプトが何をしているのかを詳しく解説します。

---

## 📚 目次

1. [train_deltaR_sty.sh - スタイル学習](#1-train_deltar_stysh---スタイル学習)
2. [train_deltaR_cnt.sh - コンテンツ学習](#2-train_deltar_cntsh---コンテンツ学習)
3. [train_QR.sh - 完全QR学習](#3-train_qrsh---完全qr学習)
4. [save_flux_residual.sh - 残差重み保存](#4-save_flux_residualsh---残差重み保存)
5. [inference_merge.sh - マージ推論](#5-inference_mergesh---マージ推論)
6. [visualize_qrlora_similarity.sh - 類似度可視化](#6-visualize_qrlora_similaritysh---類似度可視化)

---

## 1. train_deltaR_sty.sh - スタイル学習

### 🎯 目的
**スタイル特徴のみを学習する**ΔR-LoRAモデルを訓練します。

### 📋 実行コマンド
```bash
bash flux_dir/train_deltaR_sty.sh <GPU_ID> <RANK>

# 例: GPU 0番を使用、ランク64
bash flux_dir/train_deltaR_sty.sh 0 64
```

### 🔧 主要パラメータ

| パラメータ | 値 | 説明 |
|----------|-----|------|
| `TRIGGER_NAME` | `<s>` | スタイルトリガーワード |
| `INSTANCE_DIR` | `assets/sty/cat` | スタイル画像ディレクトリ（猫の画像） |
| `INSTANCE_PROMPT` | `"a cat in <s> style"` | 学習時のプロンプト |
| `VALID_PROMPT` | `"a dog in <s> style"` | 検証時のプロンプト（異なる被写体で汎化性を確認） |
| `MODEL_NAME` | `black-forest-labs/FLUX.1-dev` | ベースモデル |
| `RANK` | `64` | LoRAのランク（低ランク分解の次元数） |
| `lora_init_method` | `triu_deltaR` | **上三角制約付きΔR初期化** |
| `max_train_steps` | `1000` | 最大学習ステップ数 |
| `learning_rate` | `1e-4` | 学習率 |
| `use_zero_init` | ✓ | ΔR行列をゼロで初期化 |

### 🔬 技術詳細

#### 学習される内容
- **Q行列**: 固定（学習されない）
- **base_R行列**: 固定（学習されない）
- **delta_R行列**: **学習対象**（スタイル情報のみを保持）

#### 数式表現
```
W_style = Q × (R_base + ΔR_style) + W_res
```

#### 初期化方法: `triu_deltaR`
1. 元の重み行列 W に対してSVD分解を実行
2. 上位r個の特異値を使用してコア行列を構築
3. コア行列に対してQR分解を実行
   - Q: 直交行列 → 固定
   - R: 上三角行列 → base_R として固定
4. ΔR行列を**上三角制約付き**でゼロ初期化
5. 残差重み W_res = W - Q×R を元の重みに適用

#### 上三角制約の意味
```python
# ΔR行列は常に上三角行列の構造を保持
ΔR = [[x, x, x, x],
      [0, x, x, x],
      [0, 0, x, x],
      [0, 0, 0, x]]
```
この制約により、R行列の上三角構造が保たれ、より強い分離性が得られます。

### 📊 出力

#### 生成されるファイル
```
exps_flux/MMDD-HHMMSS-<s>-64/
├── pytorch_lora_weights.safetensors    # 最終重み
├── train_script.sh                      # 実行したスクリプトのコピー
├── checkpoint-250/                      # チェックポイント（250ステップごと）
│   ├── pytorch_lora_weights.safetensors
│   ├── optimizer.bin
│   ├── scheduler.bin
│   └── random_states_0.pkl
├── checkpoint-500/
├── checkpoint-750/
├── checkpoint-1000/
└── logs/
    └── dreambooth-flux-dev-qrlora-deltaR/
        └── events.out.tfevents.*        # TensorBoard ログ
```

#### 重みファイルの構造
```python
# pytorch_lora_weights.safetensors の内容
{
    "transformer.single_transformer_blocks.0.attn.to_q.lora.q.weight": Tensor,        # Q行列（固定）
    "transformer.single_transformer_blocks.0.attn.to_q.lora.base_r.weight": Tensor,   # base R行列（固定）
    "transformer.single_transformer_blocks.0.attn.to_q.lora.delta_r.weight": Tensor,  # ΔR行列（学習済み）
    # ... 他のレイヤーも同様
}
```

### 📈 学習進行

1. **ステップ 0-250**: 初期学習フェーズ
   - ΔR行列がゼロから学習開始
   - スタイル特徴を徐々に獲得
   
2. **ステップ 250-500**: 中期学習フェーズ
   - スタイル特徴が強化される
   - 検証画像で進捗確認
   
3. **ステップ 500-750**: 後期学習フェーズ
   - スタイル特徴が安定化
   
4. **ステップ 750-1000**: 最終調整フェーズ
   - 微調整と収束

### 🎨 期待される結果
- **入力**: 猫の画像（特定のスタイル）
- **学習内容**: そのスタイルの視覚的特徴（色調、筆触、質感など）
- **検証**: 犬の画像に同じスタイルを適用できるか確認
- **出力モデル**: スタイル情報のみを持つΔR_style行列

---

## 2. train_deltaR_cnt.sh - コンテンツ学習

### 🎯 目的
**コンテンツ特徴（形状・構造）のみを学習する**ΔR-LoRAモデルを訓練します。

### 📋 実行コマンド
```bash
bash flux_dir/train_deltaR_cnt.sh <GPU_ID> <RANK>

# 例: GPU 0番を使用、ランク64
bash flux_dir/train_deltaR_cnt.sh 0 64
```

### 🔧 主要パラメータ

| パラメータ | 値 | 説明 |
|----------|-----|------|
| `TRIGGER_NAME` | `<c>` | コンテンツトリガーワード |
| `INSTANCE_DIR` | `assets/cnt/dog` | コンテンツ画像ディレクトリ（犬の画像） |
| `INSTANCE_PROMPT` | `"a photo of a <c> dog"` | 学習時のプロンプト |
| `VALID_PROMPT` | `"a photo of a <c> dog on the beach"` | 検証時のプロンプト（異なる背景で汎化性を確認） |
| `MODEL_NAME` | `black-forest-labs/FLUX.1-dev` | ベースモデル |
| `RANK` | `64` | LoRAのランク |
| `lora_init_method` | `triu_deltaR` | 上三角制約付きΔR初期化 |
| `max_train_steps` | `1000` | 最大学習ステップ数 |
| `learning_rate` | `1e-4` | 学習率 |
| `use_zero_init` | ✓ | ΔR行列をゼロで初期化 |

### 🔬 技術詳細

#### 学習される内容
- **Q行列**: 固定（スタイル学習と同じQ）
- **base_R行列**: 固定（スタイル学習と同じbase_R）
- **delta_R行列**: **学習対象**（コンテンツ情報のみを保持）

#### 数式表現
```
W_content = Q × (R_base + ΔR_content) + W_res
```

#### スタイル学習との違い
- **学習データ**: 犬の写真（特定の個体の形状・特徴）
- **学習目標**: 物体の形状、構造、アイデンティティ
- **検証方法**: 異なる背景（ビーチ）で同じ犬を生成できるか

### 📊 出力

```
exps_flux/MMDD-HHMMSS-<c>-64/
├── pytorch_lora_weights.safetensors    # コンテンツ特徴を持つΔR_content
├── train_script.sh
├── checkpoint-250/
├── checkpoint-500/
├── checkpoint-750/
├── checkpoint-1000/
└── logs/
```

### 🎨 期待される結果
- **入力**: 特定の犬の画像
- **学習内容**: その犬の形状、顔の特徴、体型など
- **検証**: ビーチなど異なる背景でも同じ犬を生成
- **出力モデル**: コンテンツ情報のみを持つΔR_content行列

### 🔑 重要な特性
スタイル学習とコンテンツ学習は**同じQとbase_Rを共有**しますが、**異なるΔR行列**を学習します：
- ΔR_style: スタイル情報のみ
- ΔR_content: コンテンツ情報のみ

これにより、後でマージする際に干渉が最小化されます。

---

## 3. train_QR.sh - 完全QR学習

### 🎯 目的
**QとRの両方の行列を同時に学習する**完全QR-LoRAモデルを訓練します。単一タスクでの高速収束を実現します。

### 📋 実行コマンド
```bash
bash flux_dir/train_QR.sh <GPU_ID> <RANK>

# 例: GPU 0番を使用、ランク64
bash flux_dir/train_QR.sh 0 64
```

### 🔧 主要パラメータ

| パラメータ | 値 | 説明 |
|----------|-----|------|
| `TRIGGER_NAME` | `[sks]` | 一般的なトリガーワード |
| `INSTANCE_DIR` | `assets/cnt/dog` | 学習画像ディレクトリ |
| `INSTANCE_PROMPT` | `"a photo of a [sks] dog"` | 学習時のプロンプト |
| `VALID_PROMPT` | `"a photo of a [sks] dog on the beach"` | 検証時のプロンプト |
| `MODEL_NAME` | `black-forest-labs/FLUX.1-dev` | ベースモデル |
| `RANK` | `64` | LoRAのランク |
| `lora_init_method` | `qr` | **完全QR分解初期化** |
| `max_train_steps` | `1000` | 最大学習ステップ数 |
| `learning_rate` | `1e-4` | 学習率 |
| `use_zero_init` | ✗ | **使用しない**（QとR両方を学習） |

### 🔬 技術詳細

#### 学習される内容
- **Q行列**: **学習対象**（直交行列）
- **R行列**: **学習対象**（上三角行列）
- **delta_R行列**: なし

#### 数式表現
```
W_full = Q × R
```

#### 初期化方法: `qr`
1. 元の重み行列 W に対してSVD分解を実行
2. 上位r個の特異値を使用してコア行列を構築
   ```
   W_core = U_r × S_r × V_r^T
   ```
3. コア行列に対してQR分解を実行
   ```
   W_core = Q × R
   ```
4. **QとR両方を学習可能なパラメータとして初期化**
5. 元の重みを更新: W_new = W - scaling × Q × R

#### ΔR方式との違い

| 項目 | ΔR方式 | 完全QR方式 |
|-----|--------|----------|
| Q行列 | 固定 | **学習** |
| R行列 | base_R固定 + ΔR学習 | **全体を学習** |
| 用途 | マルチタスク（スタイル/コンテンツ分離） | 単一タスク |
| 収束速度 | 標準 | **高速** |
| パラメータ数 | r × d（ΔRのみ） | r × (m + d)（Q + R） |
| マージ可能性 | ✓（異なるΔRをマージ） | ✗（単一タスク用） |

### 📊 出力

```
exps_flux/MMDD-HHMMSS-[sks]-64/
├── pytorch_lora_weights.safetensors    # QとRの両方が含まれる
├── train_script.sh
├── checkpoint-250/
├── checkpoint-500/
├── checkpoint-750/
└── logs/
```

#### 重みファイルの構造
```python
# pytorch_lora_weights.safetensors の内容
{
    "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight": Tensor,  # R行列（学習済み）
    "transformer.single_transformer_blocks.0.attn.to_q.lora_B.weight": Tensor,  # Q行列（学習済み）
    # ... 他のレイヤーも同様
}
```

### 🎨 期待される結果
- **入力**: 特定の犬の画像
- **学習内容**: その犬のすべての特徴（スタイル+コンテンツ）
- **収束**: ΔR方式より高速
- **出力モデル**: 完全なQR-LoRAモデル（マージ不可）

### 💡 使用シーン
- 単一の被写体やスタイルを学習したい場合
- スタイル/コンテンツの分離が不要な場合
- より高速な収束が必要な場合
- DreamBoothのような単一タスク学習

---

## 4. save_flux_residual.sh - 残差重み保存

### 🎯 目的
**推論時の初期化オーバーヘッドを削減**するため、FLUXモデルの残差重み（W_res）を事前計算して保存します。

### 📋 実行コマンド
```bash
bash flux_dir/save_flux_residual.sh <GPU_ID>

# 例: GPU 0番を使用
bash flux_dir/save_flux_residual.sh 0
```

### 🔧 主要パラメータ

| パラメータ | 値 | 説明 |
|----------|-----|------|
| `MODEL_PATH` | `black-forest-labs/FLUX.1-dev` | ベースモデル |
| `RANK` | `64` | LoRAのランク（学習時と同じ） |
| `OUTPUT_DIR` | `flux_dir` | 出力ディレクトリ |

### 🔬 技術詳細

#### 処理の流れ

1. **FLUXモデルをロード**
   ```python
   pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
   ```

2. **対象モジュールに対してQR分解を実行**
   - 対象: アテンション層とFFN層
   ```python
   target_modules = [
       "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
       "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
       "ff.net.0.proj", "ff.net.2",
       "ff_context.net.0.proj", "ff_context.net.2",
   ]
   ```

3. **各レイヤーで残差を計算**
   ```python
   # 元の重み
   W_original = module.weight.data
   
   # SVD → QR分解
   U, S, Vh = torch.linalg.svd(W_original)
   core_matrix = U[:, :r] @ diag(S[:r]) @ Vh[:r, :]
   Q, R = torch.linalg.qr(core_matrix)
   
   # 残差を計算
   W_res = W_original - Q @ R
   ```

4. **残差重みを保存**
   ```python
   residual_dict = {
       "layer_name.residual.weight": W_res,
       # ... 全レイヤー
   }
   save_file(residual_dict, "flux_residual_weights.safetensors")
   ```

### 📊 出力

```
flux_dir/
└── flux_residual_weights.safetensors    # 約20GB
```

#### ファイル内容
```python
{
    "single_transformer_blocks.0.attn.to_q.residual.weight": Tensor[shape=(3072, 3072)],
    "single_transformer_blocks.0.attn.to_k.residual.weight": Tensor[shape=(3072, 3072)],
    # ... 全対象レイヤーの残差重み
}
```

### ⏱️ 実行時間
- **初回**: 10-15分（モデルダウンロード + QR分解）
- **2回目以降**: 5-10分（QR分解のみ）

### 💡 なぜ必要か？

#### 残差重みなしの推論
```python
# 推論のたびにQR分解を実行（遅い）
W = Q @ (R + merged_ΔR)
```

#### 残差重みありの推論
```python
# QR分解は事前計算済み（速い）
W = Q @ (R + merged_ΔR) + W_res（事前計算済み）
```

**高速化効果**: 推論開始までの時間が **5-10分短縮**

### 🔑 重要な注意点
- **ランクを変更した場合は再生成が必要**
- 学習時のランクと一致させる必要がある
- ベースモデルが異なる場合も再生成が必要

---

## 5. inference_merge.sh - マージ推論

### 🎯 目的
**スタイルとコンテンツのΔR行列をマージして画像を生成**します。異なるスケール係数の組み合わせで複数の画像を生成します。

### 📋 実行コマンド
```bash
bash flux_dir/inference_merge.sh <GPU_ID>

# 例: GPU 0番を使用
bash flux_dir/inference_merge.sh 0
```

### 🔧 主要パラメータ

| パラメータ | 値 | 説明 |
|----------|-----|------|
| `MODEL_PATH` | `black-forest-labs/FLUX.1-dev` | ベースモデル |
| `STYLE_LORA_PATH` | `exps_flux/xx/checkpoint-1000/...` | スタイルLoRA重み |
| `CONTENT_LORA_PATH` | `exps_flux/xx/checkpoint-1000/...` | コンテンツLoRA重み |
| `RESIDUAL_PATH` | `flux_dir/flux_residual_weights.safetensors` | 残差重み（事前計算） |
| `PROMPT` | `"a <c> dog in <s> style"` | 生成プロンプト |
| `NUM_STEPS` | `28` | 拡散ステップ数 |
| `SEED` | `42` | ランダムシード |
| `STYLE_WEIGHTS` | `"0.9,1.0"` | スタイルのスケール係数（カンマ区切り） |
| `CONTENT_WEIGHTS` | `"0.9,1.0"` | コンテンツのスケール係数（カンマ区切り） |
| `DTYPE` | `fp16` | 推論時のデータ型 |

### 🔬 技術詳細

#### マージの数式
```python
# 各レイヤーで以下の計算を実行
merged_ΔR = α × ΔR_style + β × ΔR_content

# 最終的な重み
W_final = Q @ (R_base + merged_ΔR) + W_res
```

ここで：
- `α`: スタイルのスケール係数（0.9, 1.0など）
- `β`: コンテンツのスケール係数（0.9, 1.0など）
- `Q`: 直交行列（スタイルとコンテンツで共有）
- `R_base`: 基底上三角行列（スタイルとコンテンツで共有）
- `W_res`: 残差重み（事前計算済み）

#### 処理の流れ

1. **モデルと重みをロード**
   ```python
   pipe = FluxPipeline.from_pretrained(MODEL_PATH)
   style_lora = load_file(STYLE_LORA_PATH)
   content_lora = load_file(CONTENT_LORA_PATH)
   residual = load_file(RESIDUAL_PATH)
   ```

2. **スケール係数の全組み合わせで画像生成**
   ```python
   for α in [0.9, 1.0]:
       for β in [0.9, 1.0]:
           # 重みをマージ
           update_model_weights(pipe, style_lora, content_lora, residual, α, β)
           
           # 画像生成
           image = pipe(prompt, num_inference_steps=28)
           
           # 保存
           image.save(f"inference_s{α:.2f}_c{β:.2f}.png")
   ```

3. **各組み合わせで生成**
   - `(α=0.9, β=0.9)`: スタイル弱め、コンテンツ弱め
   - `(α=0.9, β=1.0)`: スタイル弱め、コンテンツ標準
   - `(α=1.0, β=0.9)`: スタイル標準、コンテンツ弱め
   - `(α=1.0, β=1.0)`: スタイル標準、コンテンツ標準

### 📊 出力

```
outputs_infer_flux/YYYYMMDD_HHMMSS/
├── script.sh                           # 実行したスクリプトのコピー
├── inference_s0.90_c0.90.png          # α=0.9, β=0.9
├── inference_s0.90_c1.00.png          # α=0.9, β=1.0
├── inference_s1.00_c0.90.png          # α=1.0, β=0.9
└── inference_s1.00_c1.00.png          # α=1.0, β=1.0
```

### 🎨 生成例の解釈

#### プロンプト: `"a <c> dog in <s> style"`
- `<c>`: コンテンツトリガー（特定の犬）
- `<s>`: スタイルトリガー（特定の画風）

#### 期待される結果
1. **inference_s1.00_c1.00.png**:
   - 学習した犬の形状・特徴を完全に反映
   - 学習した画風を完全に反映
   - スタイルとコンテンツが最大限に統合

2. **inference_s0.90_c1.00.png**:
   - 犬の特徴は完全に保持
   - 画風は90%の強度（やや弱め）

3. **inference_s1.00_c0.90.png**:
   - 画風は完全に反映
   - 犬の特徴は90%の強度（やや汎化）

4. **inference_s0.90_c0.90.png**:
   - 両方とも90%の強度
   - より汎化的な結果

### ⏱️ 実行時間
- 1画像あたり: 約30-60秒（GPU性能による）
- 4組み合わせ合計: 約2-4分

### 💡 スケール係数の調整

#### より細かい制御
```bash
STYLE_WEIGHTS="0.7,0.8,0.9,1.0,1.1"
CONTENT_WEIGHTS="0.7,0.8,0.9,1.0,1.1"
```
この場合、5×5=**25枚の画像**が生成されます。

#### 推奨範囲
- **0.8-1.2**: 通常の調整範囲
- **0.5-0.7**: 弱い影響（汎化）
- **1.3-1.5**: 強い影響（過学習の可能性）

### 🔑 QR-LoRAの優位性

#### 従来のLoRAマージ
```python
# 線形補間（干渉あり）
W_merged = α × W_lora1 + β × W_lora2
```
問題: 重み空間で直接マージすると干渉が発生

#### QR-LoRAマージ
```python
# ΔR空間でのマージ（干渉最小）
merged_ΔR = α × ΔR_style + β × ΔR_content
W_final = Q @ (R + merged_ΔR) + W_res
```
利点:
- **直交性**: Qが直交行列なので干渉が最小化
- **加法性**: ΔRの要素ごとの加算で簡単にマージ
- **分離性**: スタイルとコンテンツが独立

---

## 6. visualize_qrlora_similarity.sh - 類似度可視化

### 🎯 目的
**2つのLoRA重みファイル間の類似度を解析・可視化**して、分離性（disentanglement）を評価します。

### 📋 実行コマンド
```bash
bash test/visualize_qrlora_similarity.sh <GPU_ID>

# 例: GPU 0番を使用
bash test/visualize_qrlora_similarity.sh 0
```

### 🔧 主要パラメータ（編集が必要）

スクリプト内で以下を設定：

```bash
# 比較するLoRAのパス
LORA1_PATH="exps_flux/xx/checkpoint-1000/pytorch_lora_weights.safetensors"
LORA2_PATH="exps_flux/xx/checkpoint-1000/pytorch_lora_weights.safetensors"

# 名前（グラフに表示）
LORA1_NAME="sty"    # スタイル
LORA2_NAME="cnt"    # コンテンツ

# 出力ディレクトリ
OUTPUT_DIR="output_vis/qrlora_sim-YYYYMMDD-HHMMSS"
```

### 🔬 技術詳細

#### 解析内容

3種類の行列について、層ごとのコサイン類似度を計算：

1. **Q行列の類似度**
   ```python
   similarity = cosine_similarity(Q_style, Q_content)
   ```
   期待: **高い類似度（≈1.0）**
   - Q行列はスタイルとコンテンツで共有
   - 固定されているため、ほぼ同一

2. **ΔR行列の類似度**
   ```python
   similarity = cosine_similarity(ΔR_style, ΔR_content)
   ```
   期待: **低い類似度（≈0.0）**
   - スタイルとコンテンツで異なる情報を保持
   - 分離性が高いほど類似度が低い

3. **base_R行列の類似度**
   ```python
   similarity = cosine_similarity(R_base_style, R_base_content)
   ```
   期待: **高い類似度（≈1.0）**
   - base_R行列も共有
   - 固定されているため、ほぼ同一

#### コサイン類似度の計算

```python
def cosine_similarity(v1, v2):
    """
    -1 ≤ similarity ≤ 1
    
    1.0  : 完全に同じ方向（同一）
    0.0  : 直交（完全に独立）
    -1.0 : 完全に反対方向
    """
    return (v1 · v2) / (||v1|| × ||v2||)
```

### 📊 出力

```
output_vis/qrlora_sim-YYYYMMDD-HHMMSS/
├── vis_script.sh                              # 実行したスクリプトのコピー
├── Q_similarity_fixed_scale_line.png          # Q行列の類似度（固定スケール）
├── Q_similarity_auto_scale_line.png           # Q行列の類似度（自動スケール）
├── Q_similarity_distribution.png              # Q行列の類似度分布
├── deltaR_similarity_fixed_scale_line.png     # ΔR行列の類似度（固定スケール）
├── deltaR_similarity_auto_scale_line.png      # ΔR行列の類似度（自動スケール）
├── deltaR_similarity_distribution.png         # ΔR行列の類似度分布
├── baseR_similarity_fixed_scale_line.png      # base_R行列の類似度（固定スケール）
├── baseR_similarity_auto_scale_line.png       # base_R行列の類似度（自動スケール）
├── baseR_similarity_distribution.png          # base_R行列の類似度分布
├── Q_similarity_details.json                  # Q行列の詳細データ
├── deltaR_similarity_details.json             # ΔR行列の詳細データ
├── baseR_similarity_details.json              # base_R行列の詳細データ
└── summary_statistics.txt                     # 統計サマリー
```

### 📈 グラフの種類

#### 1. 層ごとの類似度折れ線グラフ
```
Cosine Similarity of deltaR matrices
(sty vs cnt)

1.0 |                                    
0.8 |                                    
0.6 |                                    
0.4 |     ●                              
0.2 |  ●     ●     ●                     
0.0 |────●────●──────●────●──────────   Layer Index →
```
- X軸: レイヤーインデックス
- Y軸: コサイン類似度
- 各点が1つのレイヤーの類似度

#### 2. 類似度分布ヒストグラム
```
Distribution of deltaR Similarity
(sty vs cnt)

頻度
 20 |     ▇▇▇▇
 15 |     ▇▇▇▇
 10 |  ▇▇▇▇▇▇▇▇
  5 |  ▇▇▇▇▇▇▇▇▇▇
  0 |─────────────────────
     -1  0  0.5  1   Similarity →
```
- X軸: 類似度の範囲
- Y軸: 頻度（レイヤー数）

### 📄 統計サマリー例

```
=== QR-LoRA Similarity Analysis Summary ===

Q Matrix Similarity:
  Mean:   0.9985
  Median: 0.9987
  Std:    0.0012
  Min:    0.9956
  Max:    0.9998
  → Q行列はほぼ同一（期待通り）

deltaR Matrix Similarity:
  Mean:   0.0342
  Median: 0.0298
  Std:    0.0456
  Min:    -0.0123
  Max:    0.1234
  → ΔR行列は低い類似度（良好な分離性）

baseR Matrix Similarity:
  Mean:   0.9978
  Median: 0.9981
  Std:    0.0018
  Min:    0.9934
  Max:    0.9995
  → base_R行列はほぼ同一（期待通り）
```

### 🎯 結果の解釈

#### 理想的なケース
```
Q行列:      平均類似度 > 0.99  (✓ 共有されている)
ΔR行列:     平均類似度 < 0.10  (✓ 良好な分離性)
base_R行列: 平均類似度 > 0.99  (✓ 共有されている)
```

#### 問題のあるケース
```
Q行列:      平均類似度 < 0.95  (✗ 共有されていない)
ΔR行列:     平均類似度 > 0.50  (✗ 分離性が低い)
base_R行列: 平均類似度 < 0.95  (✗ 共有されていない)
```

### 💡 分離性の重要性

#### 高い分離性（ΔR類似度が低い）
- スタイルとコンテンツが独立
- マージ時に干渉が少ない
- スケール係数で柔軟に制御可能

#### 低い分離性（ΔR類似度が高い）
- スタイルとコンテンツが混在
- マージ時に意図しない干渉
- 予期しない結果が生成される可能性

### 🔍 詳細データ（JSON）

```json
{
  "layer_name": "transformer.single_transformer_blocks.0.attn.to_q",
  "similarity": 0.0342,
  "lora1_norm": 12.5678,
  "lora2_norm": 11.2345,
  "dimension": [64, 3072]
}
```

各レイヤーの詳細な数値データが保存されます。

---

## 🔄 完全なワークフロー例

実際の使用例を示します：

### ステップ1: スタイル学習（猫の画風）
```bash
bash flux_dir/train_deltaR_sty.sh 0 64
# 出力: exps_flux/1001-160000-<s>-64/pytorch_lora_weights.safetensors
```

### ステップ2: コンテンツ学習（犬の形状）
```bash
bash flux_dir/train_deltaR_cnt.sh 0 64
# 出力: exps_flux/1001-170000-<c>-64/pytorch_lora_weights.safetensors
```

### ステップ3: 残差重み保存（初回のみ）
```bash
bash flux_dir/save_flux_residual.sh 1
# 出力: flux_dir/flux_residual_weights.safetensors
```

### ステップ4: 類似度解析（オプション）
```bash
# スクリプトを編集
vi test/visualize_qrlora_similarity.sh
# LORA1_PATH と LORA2_PATH を設定

bash test/visualize_qrlora_similarity.sh 1
# 出力: output_vis/qrlora_sim-YYYYMMDD-HHMMSS/
```

### ステップ5: マージ推論
```bash
# スクリプトを編集
vi flux_dir/inference_merge.sh
# STYLE_LORA_PATH と CONTENT_LORA_PATH を設定

bash flux_dir/inference_merge.sh 1
# 出力: outputs_infer_flux/YYYYMMDD_HHMMSS/*.png
```

### 最終結果
- 猫の画風で描かれた犬の画像
- スタイルとコンテンツを独立して制御可能
- 複数のスケール係数で微調整された画像

---

## 🎓 まとめ

### 各スクリプトの役割

| スクリプト | 役割 | 出力 | 所要時間 |
|----------|------|------|---------|
| `train_deltaR_sty.sh` | スタイル学習 | ΔR_style | 2-4時間 |
| `train_deltaR_cnt.sh` | コンテンツ学習 | ΔR_content | 2-4時間 |
| `train_QR.sh` | 完全QR学習 | Q + R | 1-3時間 |
| `save_flux_residual.sh` | 残差重み保存 | W_res | 5-15分 |
| `inference_merge.sh` | マージ推論 | 画像 | 2-4分 |
| `visualize_qrlora_similarity.sh` | 類似度解析 | グラフ・統計 | 1-2分 |

### QR-LoRAの核心

1. **QR分解**: W = Q × R
2. **固定と学習**: Qを固定、ΔRのみ学習
3. **分離**: スタイルとコンテンツが独立したΔR
4. **マージ**: ΔR空間での加算（干渉最小）
5. **制御**: スケール係数で柔軟に調整

### 理論的基盤

```
W_final = Q × (R_base + α·ΔR_style + β·ΔR_content) + W_res

ここで:
- Q: 直交行列（固定） → 干渉を最小化
- R_base: 上三角行列（固定） → 基底表現
- ΔR_style: 学習済みスタイル変化
- ΔR_content: 学習済みコンテンツ変化
- W_res: 残差重み → 元の情報を保存
- α, β: スケール係数 → 柔軟な制御
```

この構造により、スタイルとコンテンツを**独立して制御**しながら、**高品質なマージ**を実現します。

---

**最終更新**: 2025年10月1日
