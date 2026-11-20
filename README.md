# Perbandingan model generatif pada Fashion-MNIST

Proyek D6421 Sesi 07 membandingkan Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, dan DDPM. Lima eksperimen MLP menguji klasifikasi, transpos bobot kelas, transpos fitur, dekoder fitur beku, serta sampling Gaussian di ruang laten classifier.

Semua model generatif memakai split 56.000/7.000/7.000, seed 42, dan batch 128. Checkpoint dipilih menggunakan validasi. Evaluasi ulang classifier memperoleh 6.213 dari 7.000 prediksi benar (88,76%). FID pada 7.000 citra riil dan 7.000 sampel per generator adalah DCGAN 41,30, DDPM 79,81, VAE 93,54 dan GAN 238,01. Hasil tersebut berlaku untuk konfigurasi dan seed yang tercatat.

| Folder | Isi |
|---|---|
| `FINAL_SOURCE_CODE` | Model, pelatihan, evaluasi, generator figur, serta tes |
| `FINAL_CHECKPOINTS` | Bobot terpilih, indeks split, dan statistik laten |
| `FINAL_METRICS` | JSON, CSV, kurva mentah, data provenance dan lingkungan |
| `FINAL_FIGURES` | Lima PNG dari checkpoint dan data eksperimen |

## Persiapan dan pengujian

Gunakan Python 3.10. Untuk RTX 4060, pasang wheel PyTorch CUDA resmi sebelum dependensi lainnya. Pada mesin CPU, gunakan indeks wheel CPU dan pilih perangkat CPU pada perintah berikut. Dataset resmi disiapkan oleh `prepare_data.py` dengan pemeriksaan checksum. FID mengunduh bobot Inception pada penggunaan pertama.

```sh
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
cd FINAL_SOURCE_CODE
python -m unittest discover -v
python run.py --smoke --device cpu --out runs/smoke_replay
python prepare_data.py --data data --out runs/dataset
```

## Mengulang pelatihan

```sh
python run.py --full --classifier-only --epochs 30 --device cpu --threads 2 --data data --out runs/mlp_replay
python classifier_experiments.py --classifier runs/mlp_replay --out runs/proof_replay --data data --epochs 30 --threads 2
python full_benchmark.py --phase train --data data --out runs/full_replay --classifier runs/mlp_replay/MLP.pt --models AE SparseAE VAE --device cpu --max-epochs 20 --min-epochs 3 --seconds-per-model 70 --threads 4
python full_benchmark.py --phase train --data data --out runs/full_replay --classifier runs/mlp_replay/MLP.pt --models GAN --device cpu --max-epochs 20 --min-epochs 3 --seconds-per-model 85 --threads 4
python full_benchmark.py --phase train --data data --out runs/full_replay --classifier runs/mlp_replay/MLP.pt --models DCGAN DDPM --device cuda --max-epochs 20 --min-epochs 5 --seconds-per-model 340 --threads 2
python full_benchmark.py --phase evaluate --data data --out runs/full_replay --classifier runs/mlp_replay/MLP.pt --device cuda --threads 2 --fid-count 7000
python render_final_figures.py --run runs/full_replay --proof runs/proof_replay --classifier runs/mlp_replay --data data --out ../replayed_figures --log runs/replayed_figure_verification
```

## Menggunakan checkpoint yang tersedia

Dari `FINAL_SOURCE_CODE`, salin checkpoint dan catatan run ke direktori baru sebelum menjalankan evaluasi. Perintah ini memakai laporan metrik dan checkpoint asli sebagai masukan, serta menulis ulang pengukuran hanya di salinan.

```sh
python -c "from pathlib import Path; from shutil import copytree; names=('full_budget','classifier_full','classifier_proof_full'); dst=Path('runs/published'); assert not dst.exists(); [copytree(Path('..')/kind/name,dst/name,dirs_exist_ok=True) for name in names for kind in ('FINAL_CHECKPOINTS','FINAL_METRICS')]"
python full_benchmark.py --phase evaluate --data data --out runs/published/full_budget --classifier runs/published/classifier_full/MLP.pt --device cuda --threads 2 --fid-count 7000
python render_final_figures.py --run runs/published/full_budget --proof runs/published/classifier_proof_full --classifier runs/published/classifier_full --data data --out ../replayed_figures --log runs/published/figure_verification
```

Lihat `FINAL_METRICS/environment.json` untuk perangkat dan versi asli. Anggaran waktu dapat mengubah jumlah epoch pada perangkat lain. Empat model mencapai batas 20 epoch, sehingga konvergensi belum ditetapkan. FID memakai Inception-v3 2.048 fitur dengan kanal abu-abu disalin ke RGB. Perbandingan menggunakan satu seed dan estimasi FID berukuran sampel terbatas.

AI membantu pengembangan kode PyTorch, debugging, pelaksanaan eksperimen, serta draf analisis dan laporan. Bantuan tersebut melampaui batas referensi sintaks dan dokumentasi dalam panduan tugas.
