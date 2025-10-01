# QR-LoRA プロジェクト概要

## 📋 プロジェクト情報

**プロジェクト名**: QR-LoRA (QR Decomposition-based LoRA)  
**会議**: ICCV 2025 採択  
**論文**: [QR-LoRA: Efficient and Disentangled Fine-tuning via QR Decomposition for Customized Generation](https://arxiv.org/abs/2507.04599v2)

### 概要
QR-LoRAは、QR分解を活用した新しいファインチューニング手法です。視覚属性を効果的に分離する構造化されたパラメータ更新を実現します。直交行列Qが異なる視覚特徴間の干渉を最小限に抑え、上三角行列Rが属性固有の変換を効率的にエンコードします。本手法では、QとR行列を固定し、追加のタスク固有ΔR行列のみを学習します。

### 主な特徴
- **優れた分離性能**: 直交分解によるコンテンツ-スタイル融合タスクで優れた分離性を実現
- **パラメータ効率**: 従来のLoRA手法の半分のパラメータで学習可能
- **簡単な統合**: 要素ごとの加算による複数の適応のマージが可能で、相互汚染なし
- **高速収束**: QとR両方の行列を学習する際の初期化戦略により高速収束を実現
- **柔軟な制御**: スケーリング係数によるコンテンツとスタイル特徴の細かい制御

---

## 📁 ディレクトリ構造とファイル説明

### ルートディレクトリ

#### `README.md`
プロジェクトの公式ドキュメント。インストール方法、クイックスタート、主な機能、謝辞、引用情報などが記載されています。

### `assets/` - データセットディレクトリ

サンプル画像とプロジェクト図を格納：
- `qr-method.png`: QR-LoRAの手法を説明する図
- `cnt/dog/cnt_dog.jpg`: コンテンツ学習用のサンプル画像（犬）
- `sty/cat/sty_cat.jpeg`: スタイル学習用のサンプル画像（猫）

### `train_scripts/` - 学習スクリプトディレクトリ

#### `custom_qr_lora.py`
**機能**: QR-LoRAの基本実装  
**内容**:
- `CustomQRLoraLayer`: QR分解に基づくLoRAレイヤーの実装
  - QR分解による初期化（`qr_init_weights`メソッド）
  - SVD分解 → QR分解の流れで重みを初期化
  - Q行列（直交行列）とR行列（上三角行列）の両方を学習
- `inject_custom_qrlora_layer`: モデルに対してカスタムQR-LoRAレイヤーを注入する関数

#### `custom_delta_r_lora.py`
**機能**: ΔR（デルタR）方式のLoRA実装  
**内容**:
- `DeltaRLoraLayer`: Q行列を固定し、ΔR行列のみを学習するレイヤー
  - Q行列は`frozen_Q`として保持（学習されない）
  - `base_R`（基底R行列）を固定し、`delta_R`のみを学習
  - コンテンツまたはスタイルの一方の特徴のみを学習する場合に使用
- `inject_delta_r_lora_layer`: モデルへのΔR-LoRAレイヤーの注入関数

#### `custom_delta_r_triu_lora.py`
**機能**: 上三角制約付きΔR-LoRA実装  
**内容**:
- `DeltaRTriuLoraLayer`: `DeltaRLoraLayer`に上三角制約を追加
  - `triu_mask`により、ΔR行列を上三角行列に制約
  - `enforce_triangular()`メソッドで常に上三角性を保証
  - R行列の上三角構造を保持しながら学習
- `inject_delta_r_triu_lora_layer`: 上三角制約付きレイヤーの注入関数

#### `train_qrlora_flux_deltaR.py`
**機能**: FLUX-dev1モデルを用いたΔR-LoRAの学習スクリプト  
**内容**:
- `DeltaRLoraLayer`または`DeltaRTriuLoraLayer`を使用
- コンテンツまたはスタイルのいずれかの特徴を学習
- AccelerateとPEFTを使用した分散学習対応
- Validationとcheckpoint保存機能
- 学習対象モジュール：アテンション層（`attn.to_k`, `attn.to_q`, `attn.to_v`など）とFFN層

#### `train_qrlora_flux_QR.py`
**機能**: FLUX-dev1モデルを用いた完全QR-LoRAの学習スクリプト  
**内容**:
- `CustomQRLoraLayer`を使用
- QとR両方の行列を同時に学習
- 単一タスクで高速収縮を実現する場合に使用
- AccelerateとPEFTを使用した分散学習対応

### `flux_dir/` - FLUX推論・保存スクリプトディレクトリ

#### `train_deltaR_cnt.sh`
**機能**: コンテンツ学習用のシェルスクリプト  
**パラメータ**:
- GPU ID（$1）とランク（$2）を引数で指定
- トリガー名: `<c>` (content)
- インスタンスディレクトリ: `assets/cnt/dog`
- プロンプト: `"a photo of a <c> dog"`
- 学習ステップ: 1000
- 初期化方法: `triu_deltaR`（上三角制約付きΔR）

#### `train_deltaR_sty.sh`
**機能**: スタイル学習用のシェルスクリプト  
**パラメータ**:
- GPU ID（$1）とランク（$2）を引数で指定
- トリガー名: `<s>` (style)
- インスタンスディレクトリ: `assets/sty/cat`
- プロンプト: `"a cat in <s> style"`
- 学習ステップ: 1000
- 初期化方法: `triu_deltaR`（上三角制約付きΔR）

#### `train_QR.sh`
**機能**: 完全QR-LoRA学習用のシェルスクリプト  
**パラメータ**:
- GPU ID（$1）とランク（$2）を引数で指定
- トリガー名: `[sks]`
- インスタンスディレクトリ: `assets/cnt/dog`
- 初期化方法: `qr`（完全QR分解）
- 単一タスクでQとRの両方を学習

#### `save_flux_residual.py`
**機能**: FLUX初期化分解行列の事前保存スクリプト  
**内容**:
- FLUXモデルから残差重み（W_res = W_original - Q@R）を計算して保存
- 推論時の初期化分解オーバーヘッドを削減
- 出力: `flux_residual_weights.safetensors`

#### `save_flux_residual.sh`
**機能**: `save_flux_residual.py`を実行するシェルスクリプト  
**パラメータ**:
- GPU ID（$1）を指定
- モデルパス、ランク（デフォルト64）、出力ディレクトリを設定

#### `inference_merge_residual.py`
**機能**: 複数のLoRA重みをマージして推論を実行  
**内容**:
- スタイルとコンテンツのΔR行列をスケール係数で加重マージ
- 公式: `W = Q(R + α·ΔR_style + β·ΔR_content) + W_res`
- 複数のスケール組み合わせで画像生成が可能
- スタイルとコンテンツの独立した制御を実現

#### `inference_merge.sh`
**機能**: `inference_merge_residual.py`を実行するシェルスクリプト  
**パラメータ**:
- コンテンツ・スタイルLoRAのパス
- 残差重みのパス（`flux_residual_weights.safetensors`）
- プロンプト: `"a <c> dog in <s> style"`
- スタイル・コンテンツの重み（カンマ区切りで複数指定可能）
- 推論ステップ数、シード値など

### `test/` - テスト・可視化ディレクトリ

#### `visualize_qrlora_similarity.py`
**機能**: 2つのΔR-LoRA重みファイル間の類似度解析と可視化  
**内容**:
- コサイン類似度の計算
- Q行列、ΔR行列、base R行列の類似度を個別に分析
- 可視化：
  - 層ごとの類似度折れ線グラフ
  - 類似度分布のヒストグラム
- 統計情報（平均、中央値、標準偏差、最小・最大値）の出力
- JSON形式での詳細データ保存

#### `visualize_qrlora_similarity.sh`
**機能**: `visualize_qrlora_similarity.py`を実行するシェルスクリプト  
**パラメータ**:
- GPU ID（$1）を指定
- LoRA1, LoRA2のパスと名前
- 出力ディレクトリ
- `--fixed_scale`オプション: 類似度軸を0-1に固定

---

## 🔄 ワークフロー

### 1. 環境セットアップ
```bash
conda create -n qrlora python=3.10 -y
conda activate qrlora
pip install -r requirements.txt
```

### 2. モデル学習

#### スタイル学習
```bash
bash flux_dir/train_deltaR_sty.sh 0 64
```
- スタイル特徴のみを学習するΔR行列を生成

#### コンテンツ学習
```bash
bash flux_dir/train_deltaR_cnt.sh 0 64
```
- コンテンツ特徴のみを学習するΔR行列を生成

#### 完全QR学習（高速収束）
```bash
bash flux_dir/train_QR.sh 0 64
```
- QとRの両方を学習（単一タスク用）

### 3. 残差重みの事前保存
```bash
bash flux_dir/save_flux_residual.sh 0
```
- 推論高速化のため、初期化分解行列を事前計算して保存

### 4. 推論（マージと生成）
```bash
bash flux_dir/inference_merge.sh 0
```
- スタイルとコンテンツのΔR行列をマージして画像生成
- 異なるスケール係数の組み合わせで複数の画像を生成

### 5. 類似度解析
```bash
bash test/visualize_qrlora_similarity.sh 0
```
- 2つのLoRA重み間の分離性を可視化・評価

---

## 🎯 主要コンセプト

### QR分解とは
行列Wを直交行列Q（Q^T Q = I）と上三角行列Rに分解：
```
W = Q × R
```

### QR-LoRAの3つのバリエーション

#### 1. 完全QR-LoRA (`CustomQRLoraLayer`)
- QとRの両方を学習
- 単一タスクで使用
- 高速収束が可能

#### 2. ΔR-LoRA (`DeltaRLoraLayer`)
- Qを固定、ΔRのみを学習
- 公式: `W = Q(R + ΔR) + W_res`
- コンテンツまたはスタイルのどちらか一方を学習

#### 3. 上三角制約付きΔR-LoRA (`DeltaRTriuLoraLayer`)
- Qを固定、ΔRを学習（上三角制約付き）
- R行列の上三角構造を保持
- より強い制約による分離性向上

### マージ方式
```
W_merged = Q(R + α·ΔR_style + β·ΔR_content) + W_res
```
- α, β: スケール係数（通常0.9〜1.0）
- 要素ごとの加算で簡単にマージ可能
- 相互汚染が少ない

---

## 📊 利点と制限

### 利点
1. **パラメータ効率**: 従来のLoRAの半分のパラメータ
2. **優れた分離性**: 直交性によるコンテンツ・スタイルの分離
3. **簡単なマージ**: 要素ごとの加算のみ
4. **柔軟な制御**: スケール係数で微調整可能

### 制限
論文にも記載されているように、分離性はLoRAマージの十分条件でも必要条件でもありません：
- 良いマージ結果は分離性を示唆できますが
- 分離性があっても必ずしも良いマージ性能を保証するわけではありません

---

## 🔧 技術詳細

### 学習対象モジュール
FLUX Transformerの以下のモジュールに適用：
- アテンション層:
  - `attn.to_k`, `attn.to_q`, `attn.to_v`, `attn.to_out.0`
  - `attn.add_k_proj`, `attn.add_q_proj`, `attn.add_v_proj`, `attn.to_add_out`
- Feed-Forward層:
  - `ff.net.0.proj`, `ff.net.2`
  - `ff_context.net.0.proj`, `ff_context.net.2`

### ハイパーパラメータ
- **ランク (r)**: 通常64（パラメータ効率と性能のバランス）
- **alpha (α)**: 通常64（スケーリング係数 = α/r = 1.0）
- **学習率**: 1e-4
- **学習ステップ**: 1000
- **バッチサイズ**: 1（勾配累積4）
- **解像度**: 512×512

---

## 📚 参考文献と謝辞

本プロジェクトは以下の研究に触発されています：
- **ZipLoRA**: 複数LoRAのマージ手法
- **B-LoRA**: バランスドLoRAアプローチ
- **PiSSA**: Principal Singular values and Singular vectors Adaptation
- **HydraLoRA**: マルチタスクLoRA

使用ライブラリ：
- HuggingFace `diffusers`, `transformers`, `accelerate`
- PyTorch ecosystem

---

## 📝 引用

```bibtex
@inproceedings{yang2025qrlora,
  title={QR-LoRA: Efficient and Disentangled Fine-tuning via QR Decomposition for Customized Generation},
  author={Jiahui Yang and Yongjia Ma and Donglin Di and Hao Li and Wei Chen and Yan Xie and Jianxun Cui and Xun Yang and Wangmeng Zuo},
  booktitle=International Conference on Computer Vision,
  year={2025}
}
```

---

**最終更新**: 2025年10月1日
