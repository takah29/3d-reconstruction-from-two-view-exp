# 3D Reconstruction from Two View

Experimental code for 3D reconstruction from 2 images

## Environment setting

```bash
pip install pipenv  # If pipenv is not installed
cd <repository root path>
pipenv sync
```

## Usage

```bash
pipenv run python 3d_reconstruction_from_two_view.py
```

### Merton College I

![corresponding_points]("./../asset/corresponding_points1.png)
![reconstruction]("./../asset/reconstruction1.png)

### Merton College III

![corresponding_points]("./../asset/corresponding_points2.png)
![reconstruction]("./../asset/reconstruction2.png)

## Reference

- 金谷健一, 菅谷保之, 金澤靖. 3 次元コンピュータビジョン計算ハンドブック. 2016.
- 佐藤淳. コンピュータビジョン - 視覚の幾何学 -. 1999.
- Multi-view Data
  - https://www.robots.ox.ac.uk/~vgg/data/mview/
