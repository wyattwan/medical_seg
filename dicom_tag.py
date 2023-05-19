import pydicom
import os


def get_dicom_tag(dicom_path):
    tags = []
    ds = pydicom.dcmread(dicom_path)

    patient_sex = ds.PatientSex
    kvp = ds.KVP
    tube_current = ds.XRayTubeCurrent
    patient_birthdate = ds.PatientBirthDate
    patient_age = 2023 - int(patient_birthdate[0:4])
    pixel_spacing = ds.PixelSpacing
    resolution = (1/pixel_spacing[0], 1/pixel_spacing[1])

    tags.append(patient_sex)
    tags.append(patient_age)
    tags.append(kvp)
    tags.append(tube_current)
    tags.append(resolution)
    return tags


if __name__ == '__main__':
    patient_id = os.listdir('./dataset/mimics_patient/')
    patient_dicom_tags = []
    male_numbers = 0
    famale_numbers = 0
    age_list = []
    resolution_list = []
    resolution_list_set = set()
    kvp_set = set()
    tube_current_set = set()
    for i in range(len(patient_id)):
        patient_id_dicom = os.listdir(
            './dataset/mimics_patient/' + patient_id[i] + '/dicom_origin/')
        dicom_path = os.path.join(
            './dataset/mimics_patient/' + patient_id[i] + '/dicom_origin/', patient_id_dicom[0])
        patient_dicom_tags = get_dicom_tag(dicom_path)
        male_numbers += (patient_dicom_tags[0] == 'M')
        famale_numbers += (patient_dicom_tags[0] == 'F')
        age_list.append(patient_dicom_tags[1])
        kvp_set.add(patient_dicom_tags[2])
        tube_current_set.add(patient_dicom_tags[3])
        resolution_list.append(patient_dicom_tags[4])
        resolution_list_set.add(patient_dicom_tags[4])

    print('male_numbers: ', male_numbers)
    print('famale_numbers: ', famale_numbers)
    print('age_min: %d, age_max: %d, age_mean: %f' % (min(age_list), max(age_list), sum(age_list) / len(age_list)))
    print('kvp_min: , kvp_max: ', min(kvp_set), max(kvp_set))
    print('tube_current_min, tube_current_max', min(tube_current_set), max(tube_current_set))
    # print('resolution_list_set', resolution_list_set)
    # print('resolution_min, resolution_max', min(resolution_list), max(resolution_list))
    print('4 mm resolution numbers: ', resolution_list.count((4.0, 4.0)))
    print('5.0 mm resolution numbers: ', resolution_list.count((5.0, 5.0)))
    print('6.25 mm resolution numbers: ', resolution_list.count((6.25, 6.25)))
