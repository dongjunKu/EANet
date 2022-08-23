import pickle
import sys
import os

if True:
    sys.path.append(os.path.dirname(
        os.path.abspath(os.path.dirname(__file__))))
    import params

p = params.Params()

data_dir = p.MYDATA_PATH

paths_train_img_left = []
paths_train_img_right = []
paths_train_disp_left = []
paths_train_disp_right = []
paths_test_img_left = []
paths_test_img_right = []

# mydata
for root, dirs, files in os.walk(data_dir + 'training'):
    for file in sorted(files):
        temp = file.split('.')
        if temp[-1] == 'png':
            if temp[0] == 'im0':
                paths_train_img_left.append(os.path.join(root, file))
            if temp[0] == 'im1':
                paths_train_img_right.append(os.path.join(root, file))
for root, dirs, files in os.walk(data_dir + 'training'):
    for file in sorted(files):
        temp = file.split('.')
        if temp[-1] == 'pfm':
            if temp[0] == 'im0':
                paths_train_disp_left.append(os.path.join(root, file))
            if temp[0] == 'im1':
                paths_train_disp_right.append(os.path.join(root, file))
for root, dirs, files in os.walk(data_dir + 'test'):
    for file in sorted(files):
        temp = file.split('.')
        if temp[-1] == 'png':
            if temp[0] == 'im0':
                paths_test_img_left.append(os.path.join(root, file))
            if temp[0] == 'im1':
                paths_test_img_right.append(os.path.join(root, file))

fout_paths_train_img_left = open(
    p.MYDATA_PATH + "paths_train_img_left.pkl", 'wb')
fout_paths_train_img_right = open(
    p.MYDATA_PATH + "paths_train_img_right.pkl", 'wb')
fout_paths_train_disp_left = open(
    p.MYDATA_PATH + "paths_train_disp_left.pkl", 'wb')
fout_paths_train_disp_right = open(
    p.MYDATA_PATH + "paths_train_disp_right.pkl", 'wb')
fout_paths_test_img_left = open(
    p.MYDATA_PATH + "paths_test_img_left.pkl", 'wb')
fout_paths_test_img_right = open(
    p.MYDATA_PATH + "paths_test_img_right.pkl", 'wb')

pickle.dump(paths_train_img_left, fout_paths_train_img_left)
pickle.dump(paths_train_img_right, fout_paths_train_img_right)
pickle.dump(paths_train_disp_left, fout_paths_train_disp_left)
pickle.dump(paths_train_disp_right, fout_paths_train_disp_right)
pickle.dump(paths_test_img_left, fout_paths_test_img_left)
pickle.dump(paths_test_img_right, fout_paths_test_img_right)

fout_paths_train_img_left.close()
fout_paths_train_img_right.close()
fout_paths_train_disp_left.close()
fout_paths_train_disp_right.close()
fout_paths_test_img_left.close()
fout_paths_test_img_right.close()

print("the number of paths in", paths_train_img_left,
      ":", len(paths_train_img_left))
print("the number of paths in", paths_train_img_right,
      ":", len(paths_train_img_right))
print("the number of paths in", paths_train_disp_left,
      ":", len(paths_train_disp_left))
print("the number of paths in", paths_train_disp_right,
      ":", len(paths_train_disp_right))
print("the number of paths in", paths_test_img_left,
      ":", len(paths_test_img_left))
print("the number of paths in", paths_test_img_right,
      ":", len(paths_test_img_right))
