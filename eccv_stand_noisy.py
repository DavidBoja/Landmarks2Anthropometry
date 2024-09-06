import argparse
import numpy as np
import pickle
from glob import glob
import os
import pandas as pd
from tqdm import tqdm

from landmark_utils import *
from measurement_utils import *

NEW_LANDMARKS_ORDER = ['Cervicale', 'Crotch', 'Lt. ASIS', 'Lt. Acromion', 'Lt. Axilla, Ant.', 'Lt. Axilla, Post.', 'Lt. Calcaneous, Post.', 'Lt. Clavicale', 'Lt. Dactylion', 'Lt. Digit II', 'Lt. Femoral Lateral Epicn', 'Lt. Femoral Medial Epicn', 'Lt. Gonion', 'Lt. Humeral Lateral Epicn', 'Lt. Humeral Medial Epicn', 'Lt. Iliocristale', 'Lt. Infraorbitale', 'Lt. Knee Crease', 'Lt. Lateral Malleolus', 'Lt. Medial Malleolus', 'Lt. Metacarpal Phal. II', 'Lt. Metacarpal Phal. V', 'Lt. Metatarsal Phal. I', 'Lt. Metatarsal Phal. V', 'Lt. Olecranon', 'Lt. PSIS', 'Lt. Radial Styloid', 'Lt. Radiale', 'Lt. Sphyrion', 'Lt. Thelion/Bustpoint', 'Lt. Tragion', 'Lt. Trochanterion', 'Lt. Ulnar Styloid', 'Nuchale', 'Rt. ASIS', 'Rt. Acromion', 'Rt. Axilla, Ant.', 'Rt. Axilla, Post.', 'Rt. Calcaneous, Post.', 'Rt. Clavicale', 'Rt. Dactylion', 'Rt. Digit II', 'Rt. Femoral Lateral Epicn', 'Rt. Femoral Medial Epicn', 'Rt. Gonion', 'Rt. Humeral Lateral Epicn', 'Rt. Humeral Medial Epicn', 'Rt. Iliocristale', 'Rt. Infraorbitale', 'Rt. Knee Crease', 'Rt. Lateral Malleolus', 'Rt. Medial Malleolus', 'Rt. Metacarpal Phal. II', 'Rt. Metacarpal Phal. V', 'Rt. Metatarsal Phal. I', 'Rt. Metatarsal Phal. V', 'Rt. Olecranon', 'Rt. PSIS', 'Rt. Radial Styloid', 'Rt. Radiale', 'Rt. Sphyrion', 'Rt. Thelion/Bustpoint', 'Rt. Tragion', 'Rt. Trochanterion', 'Rt. Ulnar Styloid', 'Sellion', 'Substernale', 'Supramenton', 'Suprasternale', 'Waist, Preferred, Post.']

def load_model(sex):
    model_path = f"models/{sex}.pkl"
    model = pickle.load(open(model_path, "rb"))
    return model

CAESAR_GENDER_MAPPER = np.load("data/gender/CAESAR_GENDER_MAPPER.npz")

def get_caesar_gender_from_name(subj_name):

    
    if subj_name[-1] == "b":
        subj_name = subj_name[:-1] + "a"
    
    if subj_name in CAESAR_GENDER_MAPPER["names"]:
        ind = np.where(CAESAR_GENDER_MAPPER["names"] == subj_name)[0].item()
        return CAESAR_GENDER_MAPPER["genders"][ind].lower()
    else:
        return None


def estimate_measurements(model,landmarks):
    
    results = []
    for meas in MEASUREMENTS_ORDER:
        pred_meas = model[meas].predict(landmarks)
        results.append(pred_meas)
    
    return np.array(results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", 
                        type=str, 
                        help="Path to CAESAR test set in A-pose with added noise from pose-independent-anthropometry repository.")
    args = parser.parse_args()

    results = []

    CM_TO_MM = 10

    all_files = glob(os.path.join(args.dataset_path,"*.npz"))
    N = len(all_files)

    models = {"male": load_model("male"),
              "female": load_model("female")}

    all_genders = []

    for i in tqdm(range(N)):

        subj_data = np.load(all_files[i])

        subj_landmarks = dict(zip(NEW_LANDMARKS_ORDER,subj_data["landmarks_noisy"] * CM_TO_MM))
        subj_landmarks = process_landmarks(subj_landmarks, 1, True)
        subj_name = all_files[i].split("/")[-1].split(".npz")[0]
        subj_gender = get_caesar_gender_from_name(subj_name)
        all_genders.append(subj_gender)
        subj_measurements = subj_data["measurements"][np.newaxis ,...] * CM_TO_MM
       
        model = models[subj_gender.lower()]
        predicted_measurements = estimate_measurements(model, subj_landmarks)
        predicted_measurements = predicted_measurements.transpose()

        ae = np.abs(subj_measurements - predicted_measurements)
        results.append(ae)

    all_genders = np.array(all_genders)
    results = np.vstack(results)
    print(f"Evaluated on {results.shape[0]} subjects")
    results_male = results[np.where(all_genders == "male")[0]]
    results_male = np.mean(results_male,axis=0)
    results_female = results[np.where(all_genders == "female")[0]]
    results_female = np.mean(results_female,axis=0)
    results_cum = np.mean(results,axis=0)

    df_res = pd.DataFrame([results_male,results_female,results_cum],
            columns=MEASUREMENTS_ORDER,
            index=["MALE AE (mm)", "FEMALE AE (mm)","AE (mm)"]).transpose()
    df_res.loc['Average'] = df_res.mean(axis=0)
    print(df_res)
