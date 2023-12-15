
from typing import List
import json
import re
import numpy as np
from scipy.spatial.transform import Rotation as sciRot


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

NORMALIZING_LANDMARK = "Substernale"
NORMALIZING_LANDMARK_INDEX = LANDMARKS_ORDER.index("Substernale")

CAESAR_LANDMARK_MAPPING =  {
     '10th Rib Midspine': '10th Rib Midspine',
     'AUX LAND': 'AUX LAND',
     'Butt Block': 'Butt Block',
     'Cervical': 'Cervicale', # FIXED
     'Cervicale': 'Cervicale', 
     'Crotch': 'Crotch',
     'Lt. 10th Rib': 'Lt. 10th Rib',
     'Lt. ASIS': 'Lt. ASIS',
     'Lt. Acromio': 'Lt. Acromion', # FIXED
     'Lt. Acromion': 'Lt. Acromion',
     'Lt. Axilla, An': 'Lt. Axilla, Ant.', # FIXED
     'Lt. Axilla, Ant': 'Lt. Axilla, Ant.', # FIXED
     'Lt. Axilla, Post': 'Lt. Axilla, Post.', # FIXED
     'Lt. Axilla, Post.': 'Lt. Axilla, Post.',
     'Lt. Calcaneous, Post.': 'Lt. Calcaneous, Post.', 
     'Lt. Clavicale': 'Lt. Clavicale',
     'Lt. Dactylion': 'Lt. Dactylion',
     'Lt. Digit II': 'Lt. Digit II',
     'Lt. Femoral Lateral Epicn': 'Lt. Femoral Lateral Epicn',
     'Lt. Femoral Lateral Epicn ': 'Lt. Femoral Lateral Epicn', # FIXED
     'Lt. Femoral Medial Epicn': 'Lt. Femoral Medial Epicn',
     'Lt. Gonion': 'Lt. Gonion',
     'Lt. Humeral Lateral Epicn': 'Lt. Humeral Lateral Epicn',
     'Lt. Humeral Medial Epicn': 'Lt. Humeral Medial Epicn',
     'Lt. Iliocristale': 'Lt. Iliocristale',
     'Lt. Infraorbitale': 'Lt. Infraorbitale',
     'Lt. Knee Crease': 'Lt. Knee Crease',
     'Lt. Lateral Malleolus': 'Lt. Lateral Malleolus',
     'Lt. Medial Malleolu': 'Lt. Medial Malleolus', # FIXED
     'Lt. Medial Malleolus': 'Lt. Medial Malleolus',
     'Lt. Metacarpal-Phal. II': 'Lt. Metacarpal Phal. II', # FIXED
     'Lt. Metacarpal-Phal. V': 'Lt. Metacarpal Phal. V', # FIXED
     'Lt. Metatarsal-Phal. I': 'Lt. Metatarsal Phal. I', # FIXED
     'Lt. Metatarsal-Phal. V': 'Lt. Metatarsal Phal. V', # FIXED
     'Lt. Olecranon': 'Lt. Olecranon',
     'Lt. PSIS': 'Lt. PSIS',
     'Lt. Radial Styloid': 'Lt. Radial Styloid',
     'Lt. Radiale': 'Lt. Radiale',
     'Lt. Sphyrio': 'Lt. Sphyrion', # FIXED
     'Lt. Sphyrion': 'Lt. Sphyrion',
     'Lt. Thelion/Bustpoin': 'Lt. Thelion/Bustpoint', # FIXED
     'Lt. Thelion/Bustpoint': 'Lt. Thelion/Bustpoint',
     'Lt. Tragion': 'Lt. Tragion',
     'Lt. Trochanterion': 'Lt. Trochanterion',
     'Lt. Ulnar Styloid': 'Lt. Ulnar Styloid',
     'Nuchale': 'Nuchale',
     'Rt. 10th Rib': 'Rt. 10th Rib',
     'Rt. ASIS': 'Rt. ASIS',
     'Rt. Acromio': 'Rt. Acromion', # FIXED
     'Rt. Acromion': 'Rt. Acromion',
     'Rt. Axilla, An': 'Rt. Axilla, Ant.', # FIXED
     'Rt. Axilla, Ant': 'Rt. Axilla, Ant.', # FIXED
     'Rt. Axilla, Post': 'Rt. Axilla, Post.', # FIXED
     'Rt. Axilla, Post.': 'Rt. Axilla, Post.',
     'Rt. Calcaneous, Post.': 'Rt. Calcaneous, Post.',
     'Rt. Clavicale': 'Rt. Clavicale',
     'Rt. Dactylion': 'Rt. Dactylion',
     'Rt. Digit II': 'Rt. Digit II',
     'Rt. Femoral Lateral Epicn': 'Rt. Femoral Lateral Epicn',
     'Rt. Femoral Lateral Epicn ': 'Rt. Femoral Lateral Epicn', # FIXED
     'Rt. Femoral Medial Epic': 'Rt. Femoral Medial Epicn', # FIXED
     'Rt. Femoral Medial Epicn': 'Rt. Femoral Medial Epicn',
     'Rt. Gonion': 'Rt. Gonion',
     'Rt. Humeral Lateral Epicn': 'Rt. Humeral Lateral Epicn',
     'Rt. Humeral Medial Epicn': 'Rt. Humeral Medial Epicn',
     'Rt. Iliocristale': 'Rt. Iliocristale',
     'Rt. Infraorbitale': 'Rt. Infraorbitale',
     'Rt. Knee Creas': 'Rt. Knee Crease', # FIXED
     'Rt. Knee Crease': 'Rt. Knee Crease',
     'Rt. Lateral Malleolus': 'Rt. Lateral Malleolus',
     'Rt. Medial Malleolu': 'Rt. Medial Malleolus', # FIXED
     'Rt. Medial Malleolus': 'Rt. Medial Malleolus',
     'Rt. Metacarpal Phal. II': 'Rt. Metacarpal Phal. II',
     'Rt. Metacarpal-Phal. V': 'Rt. Metacarpal Phal. V', # FIXED
     'Rt. Metatarsal-Phal. I': 'Rt. Metatarsal Phal. I', # FIXED
     'Rt. Metatarsal-Phal. V': 'Rt. Metatarsal Phal. V', # FIXED
     'Rt. Olecranon': 'Rt. Olecranon',
     'Rt. PSIS': 'Rt. PSIS',
     'Rt. Radial Styloid': 'Rt. Radial Styloid',
     'Rt. Radiale': 'Rt. Radiale',
     'Rt. Sphyrio': 'Rt. Sphyrion', # FIXED
     'Rt. Sphyrion': 'Rt. Sphyrion',
     'Rt. Thelion/Bustpoin': 'Rt. Thelion/Bustpoint', # FIXED
     'Rt. Thelion/Bustpoint': 'Rt. Thelion/Bustpoint',
     'Rt. Tragion': 'Rt. Tragion',
     'Rt. Trochanterion': 'Rt. Trochanterion',
     'Rt. Ulnar Styloid': 'Rt. Ulnar Styloid',
     'Sellion': 'Sellion',
     'Substernale': 'Substernale',
     'Supramenton': 'Supramenton',
     'Suprasternale': 'Suprasternale',
     'Waist, Preferred, Post.': 'Waist, Preferred, Post.'
    }

def parse_landmark_txt_coords_formatting(data: List[str]):
    """
    Parse landamrk txt file with formatting
    x y z landmark_name

    :param data (List[str]) list of strings, each string
                represents one line from the txt file
    
    :return landmarks (dict) in formatting 
                      {landmark_name: [x,y,z]}
    """

    # get number of landmarks
    N = len(data)
    if data[-1] == "\n":
        N -= 1

    # define landmarks
    landmarks = {}

    for i in range(N):
        splitted_line = data[i].split(" ")
        x = float(splitted_line[0])
        y = float(splitted_line[1])
        z = float(splitted_line[2])

        remaining_line = splitted_line[3:]
        landmark_name = " ".join(remaining_line)
        if landmark_name[-1:] == "\n":
            landmark_name = landmark_name[:-1]

        landmarks[landmark_name] = [x,y,z]

    return landmarks

def process_caesar_landmarks(landmark_path: str, scale: float = 1000.0):
    """
    Process landmarks from .lnd file - reading file from AUX to END flags.

    :param landmark_path (str): path to landmark .lnd file
    :param scale (float): scale of landmark coordinates

    :return landmark_dict (dict): dictionary with landmark names as keys and
                                    landmark coordinates as values
                                    landmark_coords are (np.array): (1,3) array 
                                    of landmark coordinates
    """

    landmark_coords = []
    landmark_names = []

    with open(landmark_path, 'r') as file:
        do_read = False
        for line in file:

            # start reading file when encounter AUX flag
            if line == "AUX =\n":
                do_read = True
                # skip to the next line
                continue
                
            # stop reading file when encounter END flag
            if line == "END =\n":
                do_read = False
            

            if do_read:
                # EXAMPLE OF LINE IN LANDMARKS
                # 1   0   1   43.22   19.77  -38.43  522.00 Sellion
                # where the coords should be
                # 0.01977, -0.03843, 0.522
                # this means that the last three floats before 
                # the name of the landmark are the coords
                
                # find landmark coordinates
                landmark_coordinate = re.findall(r"[-+]?\d+\.*\d*", line)
                # print(line)
                # print(landmark_coordinate)
                x = float(landmark_coordinate[-3]) / scale
                y = float(landmark_coordinate[-2]) / scale
                z = float(landmark_coordinate[-1]) / scale
                
                # find landmark name
                # (?: ......)+ repeats the pattern inside the parenthesis
                # \d* says it can be 0 or more digits in the beginning
                # [a-zA-Z]+ says it needs to be one or more characters
                # [.,/]* says it can be 0 or more symbols
                # \s* says it can be 0 ore more spaces
                # NOTE: this regex misses the case for landmarks with names
                # AUX LAND 79 -- it parses it as AUX LAND -- which is ok for our purposes
                landmark_name = re.findall(r" (?:\d*[a-zA-Z]+[-.,/]*\s*)+", line)
                landmark_name = landmark_name[0][1:-1]
                landmark_name_standardized = CAESAR_LANDMARK_MAPPING[landmark_name]

                # * zero or more of the preceding character. 
                # + one or more of the preceding character.
                # ? zero or one of the preceding character.
                
                landmark_coords.append([x,y,z])
                landmark_names.append(landmark_name_standardized)
                
    landmark_coords = np.array(landmark_coords)

    return dict(zip(landmark_names, landmark_coords))

def load_landmarks(path: str):
    """
    Load landmarks from file and return the landmarks as
    np array of dimension 73 x 3 which corresponds to 
    landmarks defined in  in LANDMARKS_ORDER.

    Landmark file is defined in the following format:
    .txt extension with each line "x y z landmark_name"
    .json extension with {landmark_name: [x,y,z]}
    .lnd extension (as defined in CAESAR)

    
    :param path: (str) of path to landmark file

    :return landmarks: np.array of landmarks 
                       with dim (K,3)
    """

    ext = path.split(".")[-1]
    supported_extensions = [".txt",".json",".lnd"]

    if ext == "txt":
        with open(path, 'r') as file:
            data = file.readlines()

        landmarks = parse_landmark_txt_coords_formatting(data)

    elif ext == "json":
        with open(path,"r") as f:
            landmarks = json.load(f)

    elif ext == "lnd":
        print("Be aware that the .lnd extension assumes you are using the CAESAR dataset.")
        landmarks = process_caesar_landmarks(path,1000)

    else:
        supported_extensions_str = ', '.join(supported_extensions)
        msg = f"Landmark extensions supported: {supported_extensions_str}. Got .{ext}."
        raise ValueError(msg)

    return landmarks

def process_landmarks(landmarks, scale, normalize_viewpoint=False):
    """
    Models expect np.array of dim (1,216) which correspond to 72 landmarks (72*3=216)
    These are obtained by centering the 73 landmark coords on the NORMALIZING_LANDMARK
    and then deleting the NORMALIZING_LANDMARK --> therefore obtaning 72 landmarks

    Create numpy array from landmarks dict of 73 landmarks

    :param landmarks (dict): {landmark_name: landmark_coordinates} where
                             landmark_name are the 73 landmarks from LANDMARKS_ORDER and
                             landmark_coordinates is a list of coords [x,y,z]
    :param scale (int): scale landmarks to mm

    :return processed_landmarks (np.array) dim (1,216)
    """

    processed_landmarks = np.zeros((73,3)) 
    for i,lm_name in enumerate(LANDMARKS_ORDER):
        processed_landmarks[i,:] = landmarks[lm_name]

    processed_landmarks = processed_landmarks - processed_landmarks[NORMALIZING_LANDMARK_INDEX,:]
    processed_landmarks = np.delete(processed_landmarks, NORMALIZING_LANDMARK_INDEX, axis=0) # (72,3)
    processed_landmarks = processed_landmarks * scale # to mm

    if normalize_viewpoint:
        demo_lm = np.load("data/demo_landmarks_processed.npy") # (72,3) in mm

        # align viewpoints
        rot = sciRot.align_vectors(demo_lm, processed_landmarks)
        processed_landmarks = rot[0].apply(processed_landmarks)

    processed_landmarks = processed_landmarks.reshape(1,-1) # (1,216)

    return processed_landmarks
