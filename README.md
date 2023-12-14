# Direct 3D Body Measurement Estimation from Sparse Landmarks

This Github presents the code for the following paper: "Direct 3D Body Measurement Estimation from Sparse Landmarks" presented at VISAPP 2024.

<p align="center">
  <img src="https://github.com/DavidBoja/Landmarks2Anthropometry/blob/main/assets/main_image.png" width="950">
</p>

<b> TL;DR :</b> Estimate 11 body measurements from 73 body landmarks with the accuracy of SOA methods. 

<br>

## 🔨 Getting started

```bash
python -m venv lm2meas
source lm2meas/bin/activate
pip install -r requirements.txt
```

<br>

## 🏃 Running

```bash
python estimate_measurements.py --landmarks_path <path-to-landmarks>
                                --sex <subject-sex>
                                --scale <scale-landmarks-to-unit>
```

where
- landmarks_path is the path to the landmarks file. See [Landmarks](#Landmarks) for possible file extensions.
- sex is the subject sex: `male` or `female`. This decides the model to use.
- scale is the unit of measurement to scale to, the landmark coordinates are multiplied with the scale

⚠️ the landmarks need to be in millimeters. Check [here](#Scale) ⚠️ 

<br>

## 🧍 Landmarks

The 73 landmarks are:

```python
LANDMARKS_ORDER = ['10th Rib Midspine',
                    'Cervicale',
                    'Crotch',
                    'Lt. 10th Rib',
                    'Lt. ASIS',
                    'Lt. Acromion',
                    'Lt. Axilla, Ant.',
                    'Lt. Axilla, Post.',
                    'Lt. Calcaneous, Post.',
                    'Lt. Clavicale',
                    'Lt. Dactylion',
                    'Lt. Digit II',
                    'Lt. Femoral Lateral Epicn',
                    'Lt. Femoral Medial Epicn',
                    'Lt. Gonion',
                    'Lt. Humeral Lateral Epicn',
                    'Lt. Humeral Medial Epicn',
                    'Lt. Iliocristale',
                    'Lt. Infraorbitale',
                    'Lt. Knee Crease',
                    'Lt. Lateral Malleolus',
                    'Lt. Medial Malleolus',
                    'Lt. Metacarpal Phal. II',
                    'Lt. Metacarpal Phal. V',
                    'Lt. Metatarsal Phal. I',
                    'Lt. Metatarsal Phal. V',
                    'Lt. Olecranon',
                    'Lt. PSIS',
                    'Lt. Radial Styloid',
                    'Lt. Radiale',
                    'Lt. Sphyrion',
                    'Lt. Thelion/Bustpoint',
                    'Lt. Tragion',
                    'Lt. Trochanterion',
                    'Lt. Ulnar Styloid',
                    'Nuchale',
                    'Rt. 10th Rib',
                    'Rt. ASIS',
                    'Rt. Acromion',
                    'Rt. Axilla, Ant.',
                    'Rt. Axilla, Post.',
                    'Rt. Calcaneous, Post.',
                    'Rt. Clavicale',
                    'Rt. Dactylion',
                    'Rt. Digit II',
                    'Rt. Femoral Lateral Epicn',
                    'Rt. Femoral Medial Epicn',
                    'Rt. Gonion',
                    'Rt. Humeral Lateral Epicn',
                    'Rt. Humeral Medial Epicn',
                    'Rt. Iliocristale',
                    'Rt. Infraorbitale',
                    'Rt. Knee Crease',
                    'Rt. Lateral Malleolus',
                    'Rt. Medial Malleolus',
                    'Rt. Metacarpal Phal. II',
                    'Rt. Metacarpal Phal. V',
                    'Rt. Metatarsal Phal. I',
                    'Rt. Metatarsal Phal. V',
                    'Rt. Olecranon',
                    'Rt. PSIS',
                    'Rt. Radial Styloid',
                    'Rt. Radiale',
                    'Rt. Sphyrion',
                    'Rt. Thelion/Bustpoint',
                    'Rt. Tragion',
                    'Rt. Trochanterion',
                    'Rt. Ulnar Styloid',
                    'Sellion',
                    'Substernale',
                    'Supramenton',
                    'Suprasternale',
                    'Waist, Preferred, Post.']
```

The possible extensions of landmark_paths are:
- `.txt` extension with each line "x y z landmark_name"
- `.json` extension with {landmark_name: [x,y,z]}
- `.lnd` extension (as defined in CAESAR where lines are  "_   _   _   _  x y z landmark_name")

<br>

## 📏 Measurements

```python
MEASUREMENTS_ORDER = ['Ankle Circumference (mm)',
                      'Arm Length (Shoulder to Elbow) (mm)',
                      'Arm Length (Shoulder to Wrist) (mm)',
                      'Arm Length (Spine to Wrist) (mm)',
                      'Chest Circumference (mm)',
                      'Crotch Height (mm)',
                      'Head Circumference (mm)',
                      'Hip Circ Max Height (mm)',
                      'Hip Circumference, Maximum (mm)',
                      'Neck Base Circumference (mm)',
                      'Stature (mm)']
```
<br>

## Scale

The input landmarks need to be defined in millimeter scale. The scale parameter multiplies the landmarks with the scale as `landmarks * scale`.

If you are unsure about the unit of the landmark coordinates, you can check the distance between landmark `Sellion` and `Lt. Calcaneous, Post.` (left heel). This should somewhat resemble to the height of the subject, from which you can infer the scale.


<br>

## 💿 Demos

You can run the `estimate_measurements.py` with the demo data:

```bash
python estimate_measurements.py --landmarks_path data/demo_landmarks.json --sex male --scale 1000
```

Since the `demo_landmarks.json` are defined in meters, we use a `scale` of 1000.

<br>

## 📝 Notes

- Landmarks need to be in millimeters
- Measurements are in millimteres
- Z-ax is the height of the person
- The model is trained on CAESAR dataset so the viewpoint of the subject landmarks should be similar to the ones in the CAESAR dataset

<br>

## 🏋🏻‍♂️ Training

We provide the names of the subjects used for training and testing the models in `data/train_test_names.npz` to facilitate comparability and repeatability. 

```
data = np.load("data/train_test_names.npz")
names_train_male = data["names_train_male"]
names_test_male = data["names_test_male"]
names_train_female = data["names_train_female"]
names_test_female = data["names_test_female"]
```

## Citation

If you use our work, please reference our paper:

```
@inproceedings{Bojanić-VISAPP24,
   title = {Direct 3D Body Measurement Estimation from Sparse Landmarks},
   author = {Bojani\'{c}, David and Bartol, Kristijan and Forest, Josep and Gumhold, Stefan and Petkovi\'{c}, Tomislav and Pribani\'{c}, Tomislav},
   booktitle={},
   year = {2024}
   url={}
}
```

## TODO

- [] Add the general viewpoint of the CAESAR examples in order to normalize out-of-sample landmarks
