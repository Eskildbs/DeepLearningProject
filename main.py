import Joblist
import Corn_segmentation

anno_path = 'annotated_images/'
test_path = 'test_images/'
output_path = 'output/'
classes = {0: 'background', 1: 'weed', 2: 'corn'}

""" Traingin new network using the given joblist """
# Loading joblist
joblist = Joblist.Joblist('Joblist_GPU', output_path)
joblist.set_parameters(log_every=1000, stopping_criteria=Corn_segmentation.No_stopping_criteria(),
                       batch_size=8, max_iter=40000, LEARNING_RATE=1E-5, weight_decay=0.001)

joblist.set_data(annotated_path=anno_path, test_path=test_path, class_dict=classes,
                 data_augmentation=True)
joblist.run(train=True, save_model=True, save_test_output=False,
            save_validation_output=False, save_filters=False)

""" Predicting using a trained model model """
# Note that this can only be done, when a model is trained or when the model is downloaded.
# The used model can be downloaded here:
# This folder ("Full_Segnet_3filter_LowDecay_LowBatch") must replace the exciting folder in "models".
# Afterwards the following code can be uncommented to segment the images placed in the folder "test_images".
# The different test images can be downloaded here: https://1drv.ms/u/s!AnbtOeX1Rnufg8xhgwIQDhv6Al73FA?e=qdD03j
# (Image DJI0268.png is the one used in the report).

#name = 'Full_Segnet_3filter_LowDecay_LowBatch'
#joblist = Joblist.Load_job_torch('models/' + name + '/', output_path, SAVE_OUTPUT=True, name=name)
#joblist.set_parameters(max_iter=1000, dynamic_regularization=True)
#joblist.set_data(annotated_path=anno_path, test_path=test_path, class_dict=classes,
#                 data_augmentation=True)
#joblist.run(train=False, save_model=False, save_test_output=True,
#            save_validation_output=False, save_filters=False, get_f1_score=False)
