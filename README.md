# 量子計算とテンソルネットワーク

- 講義資料:
  - [tnqc-1.pdf](tnqc-1.pdf): 次元の呪いと圧縮
  - [tnqc-2.pdf](tnqc-2.pdf):
  - [tnqc-3.pdf](tnqc-3.pdf):
  - [tnqc-4.pdf](tnqc-4.pdf):
  - [tnqc-5.pdf](tnqc-5.pdf):

- [data](data): 実習で使用するデータファイル

- 実習コード
  - 言語
    - [python](python): Python実習コード
    - [julia](julia): Julia実習コード
    - [rust](rust): Rust実習コード
  - Hands-onコード
    - 11_svd
    - 12_image-compression
    - 13_tensor
    - 14_contraction
    - 5_mps2statevector
    - 6_statevector2mps
    - 7_tebd
    - 8_function2qtt
    - 9_finite-difference

- Python
  - 以下、`python`ディレクトリ内での実行を想定
  - Python の仮想環境準備
    - `python3 -m venv .venv`
    - `source .venv/bin/activate`
  - 必要なパッケージのインストール
    - `pip3 install -r requirements.txt`
  - 実行例
    - `python3 11_svd.py`

- Julia
  - 以下、`julia`ディレクトリ内での実行を想定
  - 必要なパッケージのインストール
    - `julia setup.jl`
  - 実行例
    - `julia --project 11_svd.jl`
  - キャッシュを使った実行
    - `repl`を立ち上げる
      - `julia --project`
    - `repl`内での実行例
      - `include("11_svd.jl")`
    - プログラムのコンパイル結果がキャッシュされるため、2回目以降の実行が高速になる

- Rust
  - 以下、`rust`ディレクトリ内での実行を想定
  - コンパイル
    - `cargo build --release`
  - 実行例
    - `cargo run --release --bin 11_svd`
  - ソースコードは`src/bin`の下にある
