
class Config:
    def __init__(self,task):
        self.task = task
        if task == "synapse":
            self.base_dir = '/homes/lzhang/data/ssl/MALBCV/Abdomen/RawData/Training/'
            self.save_dir = 'code/synapse_data'
            self.patch_size = (64, 128, 128)
            self.num_cls = 14
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 30
            
            
        if task == "synapse_re":
            self.base_dir = '/homes/lzhang/data/ssl/MALBCV/Abdomen/RawData/Training/'
            self.save_dir = 'code/synapse_data'
            self.patch_size = (64, 128, 128)
            self.num_cls = 14
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50

        if task == 'word':

            self.base_dir = '/homes/lzhang/data/WORD-V0.1.0/ssl'
            self.save_dir = '/homes/lzhang/data/ssl/DHC/code/word_data_better'
            self.patch_size = (128, 128, 128)
            self.num_cls = 17
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50

        if task == 'chd':

            self.base_dir = '/homes/lzhang/data/contrast/positional_cl-main/dataset/dataset/CHD/better/'
            self.save_dir = 'code/chd_data'
            self.patch_size = (64, 128, 128)
            self.num_cls = 8
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50

        if task ==  'amos':
            self.base_dir = '/homes/lzhang/data/ssl/DHC/code/data/Datasets/amos22'
            self.save_dir = 'code/amos_data'
            self.patch_size = (64, 128, 128)
            self.num_cls = 16
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50
            
        if task == 'acc':
            self.base_dir = '/homes/lzhang/CT'
            self.save_dir = 'acc_data'
            self.patch_size = (512, 512, 512)
            self.num_cls = 8
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50   
            
        if task == 'acc_s':
            self.base_dir = '/homes/lzhang/CT'
            self.save_dir = '/homes/lzhang/data/ssl/dhc2/DHC/code/acc_t_data'
            self.patch_size = (64, 128, 128)
            self.num_cls = 8
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50   
        
        if task == 'covid':
            self.base_dir = '/homes/lzhang/data/ssl/COVID-19-20/COVID-19-20_v2'
            self.save_dir = '/homes/lzhang/data/ssl/dhc2/DHC/code/covid_data'
            self.patch_size = (96, 128, 128)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50
            
        if task == 'colon':
            self.base_dir = '/homes/lzhang/data/colon/labeled/nii'
            self.save_dir = '/homes/lzhang/data/ssl/dhc2/DHC/code/colon_data'
            self.patch_size = (96, 128, 128)
            self.num_cls = 3
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 30
            
        if task == 'colon_u':
            self.base_dir = '/homes/lzhang/data/colon/imagesTr'
            self.save_dir = '/homes/lzhang/data/ssl/dhc2/DHC/code/colon_u_data'
            self.patch_size = (96, 128, 128)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50


