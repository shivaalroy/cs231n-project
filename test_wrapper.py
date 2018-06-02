from wrapper import OASIS

dataset = OASIS('scans', 'OASIS3_MRID2Label_051418.csv', preload = True)

image_array, label = dataset[3]
print(image_array.shape)
