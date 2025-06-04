from Json_Functions import Json_Functions

class Unet_Config(Json_Functions): # function to write to json?
    def __init__(self, json_filepath = None, **kwargs):
        # load from json file if provided
        if json_filepath != None:
            self.load_json(json_filepath)
        else:
            self.load_default_params()
            
        # update parameters from kwargs
        self.__dict__.update(kwargs)
    
    def load_default_params(self):
        # general settings
        self.name = 'Unet'
        self.dataset_dir = '/tf/Documents/Unet/Training_sets/'
        #self.model_dir = model_dir

        #self.device_count = {'GPU': 2, 'GPU': 3, 'GPU': 4}
        self.device_count = {'GPU': 1}
        self.device_map = {
                0: 0,
                1: 3,
                2: 1,
                3: 2
            }
        self.gpu_count = len(self.device_count)
        self.visible_gpu = "3"
        
        self.use_horovod = False
        self.use_cpu = False
        
        # Tensorboard_Settings
        self.use_tensorboard = False
        self.write_graph = False
        self.write_images = False
        self.write_grads = False

        # image settings
        self.tile_size = (512, 512)
        self.padding_size = (128, 128)
        self.image_channel = 1
        self.input_size = self.tile_size + (self.image_channel,)
        self.use_binary_erosion = True
        self.use_binary_dilation = False
        self.disk_size = 1

        # augmentation settings
        self.num_augmented_images = 150
        #self.augmentations_p = 0.9
        
        
        self.random_rotate = True
        self.random_rotate_p = 0.9
        
        self.flip = True
        self.transpose = True
        
        self.blur_group = True
        self.motion_blur = True
        self.motion_blur_p = 0.7
        self.median_blur = True
        self.median_blur_limit = 3
        self.median_blur_p = 0.5
        self.blur = True
        self.blur_limit = 5
        self.blur_p = 0.6
        self.blur_group_p = 0.8
        
        self.shift_scale_rotate = True
        self.shift_limit = 0.1
        self.scale_limit = 0.1
        self.rotate_limit = 45
        self.shift_scale_rotate_p = 0.7
        
        self.distortion_group = True
        self.optical_distortion = True
        self.optical_distortion_p = 0.6
        self.elastic_transform = True
        self.elastic_transform_p = 0.5
        self.grid_distortion = True
        self.grid_distortion_p = 0.6
        self.distortion_group_p = 0.7
        
        self.brightness_contrast_group = True
        self.clahe = True
        self.random_brightness_contrast = True
        self.brightness_contrast_group_p = 0.7
        
        
        
        
        
        
        
        
        self.random_rotate = True
        self.random_rotate_p = 0.9
        
        self.flip = True
        self.transpose = True
        
        self.blur_group = False
        self.blur_group_p = 0.3
        # from true to false
        self.motion_blur = False
        self.motion_blur_p = 0.1
        self.median_blur = False
        self.median_blur_limit = 3
        self.median_blur_p = 0.3
        self.blur = False
        self.blur_limit = 3
        self.blur_p = 0.3
        
        self.shift_scale_rotate = True
        self.shift_scale_rotate_p = 0.3
        self.shift_limit = 0.0625
        self.scale_limit = 0.5
        self.rotate_limit = 45

        self.distortion_group = True
        self.distortion_group_p = 0.2
        self.optical_distortion = True
        self.optical_distortion_p = 0.3
        self.elastic_transform = True
        self.elastic_transform_p = 0.3
        self.grid_distortion = True
        self.grid_distortion_p = 0.3
        
        self.brightness_contrast_group = True
        self.brightness_contrast_group_p = 0.6
        self.clahe = True
        self.sharpen = True
        self.random_brightness_contrast = True

        ### Model parameters
        # convolution
        self.filters = 32

        # optimizer
        self.optimizer = "sgd"
        self.learning_rate = 0.001
        self.decay = 1e-6
        self.momentum = 0.9
        self.nesterov = True

        # loss functions
        self.loss = 'jaccard_distance_loss'
        self.metrics = ['binary_accuracy']
        self.num_epochs = 70
        self.val_split = 0.1
        self.batch_size_per_GPU = 2
        self.batch_size = self.batch_size_per_GPU # * self.gpu_count

        # dropout
        self.dropout_value = 0.2

        # initializer
        self.initializer = 'he_normal'