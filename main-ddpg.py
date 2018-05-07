import torch
import os
import inspect
from utils import parseArguments, initSavePath, saveCode, initGamesLogger, attachSignalsHandler
import torch.backends.cudnn as cudnn
from Results import Results

# init seeds
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

# init current file (script) folder
baseFolder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # script directory

# parse arguments
args = parseArguments()

# set CUDA device
# TODO: move args.type to settings.json
if 'cuda' in args.type:
    torch.cuda.set_device(args.gpus)
    cudnn.benchmark = True

# init save path
save_path, train_path, folderName = initSavePath(args.results_dir)

# save source code
code_path = saveCode(save_path, baseFolder)

# init Results object
results = Results(save_path, folderName)

# init games logger
gamesLogger = initGamesLogger('Games', save_path)

# log PID
gamesLogger.info('PID:[{}]'.format(os.getpid()))

# handle SIGTERM
attachSignalsHandler(results, gamesLogger)

