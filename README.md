# head_ct_classifier
Couse project EECS496 Statistical Machine Learning
CT (Computed Topography) is a widely used non-invasive diagnosis approach. Head CT (Computed Topography) is the major radiology imaging diagnositc method for head trauma and corresponding complications. Head trauma including Intraparenchymal hemorrhage, Intraventricular hemorrhage, Subdural hematoma, Extradural hematoma, Subarachnoid hematoma, cranial fracture, midline shift and mass effect.In this project, I used pre-trained and fine tuned image classification models on head CT.

Multiple preprocessing for CT are provided in this study to fit head CT into pretrained pytorch image classification models.

# Input & output
Input is a NIfTI file in nii.gz format containing one head CT scan. Output is predicted yes / no for 14 labels, including Intracranial hemorrhage, Intraparenchymal hemorrhage, Intraventricular hemorrhage, Subdural hematoma, Extradural hematoma, Subarachnoid hematoma, bleeding location-left, bleeding location-right, chronic bleeding, cranial fracture, calvarial fracture, other fracture, mass effect and midline shift.
