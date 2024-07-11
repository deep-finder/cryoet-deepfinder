from copick.impl.filesystem import CopickRootFSSpec
import deepfinder.utils.copick_tools as tools
from deepfinder.inference import Segment
import deepfinder.utils.smap as sm
import tensorflow as tf
import click

@click.group()
@click.pass_context
def cli(ctx):
    pass

@cli.command()
@click.option(    
    "--predict-config",
    type=str,
    required=True,
    help="Path to the copick config file.",)
@click.option(    
    "--path-weights",
    type=str,
    required=True,
    help="Path to the Trained Model Weights.",)
@click.option(
    "--n-class",
    type=int,
    required=True,
    help="Number of classes.",
)
@click.option(
    "--patch-size",
    type=int,
    required=True,
    help="Patch of Volume for Input to Network.",
)
@click.option(
    "--user-id", 
    type=str, 
    default='deepfinder', 
    show_default=True, 
    help="User ID filter for input."
)
@click.option(
    "--session-id", 
    type=str, 
    default=None, 
    show_default=True, 
    help="Session ID filter for input."
)
@click.option(
    "--voxel-size",
    type=float,
    required=False,
    default=10.0,
    show_default=True,
    help="Voxel size of the tomograms to segment.",
)
@click.option(
    "--tomogram-algorithm",
    type=str,
    required=False,
    default='wbp',
    show_default=True,
    help="Tomogram Algorithm.",
)
@click.option(
    "--parallel-mpi/--no-parallel-mpi",
    default=False,
    help="Patch of Volume for Input to Network.",
)
def segment(
    predict_config: str, 
    path_weights: str,
    n_class: int, 
    patch_size: int,
    user_id: str, 
    session_id: str, 
    voxel_size: float = 10,
    tomogram_algorithm: str = 'denoised', 
    parallel_mpi: bool = False,
    ):

    # Determine if Using MPI or Sequential Processing
    if parallel_mpi: 

        from mpi4py import MPI
        # Initialize MPI (Get Rank and nProc)
        comm = MPI.COMM_WORLD; rank = comm.Get_rank(); nProcess = comm.Get_size()
        locGPU = rank % len(tf.config.list_physical_devices('GPU'))
    else:
        locGPU = None
        nProcess = 1
        rank = 0

    ############## (Step 1) Initialize segmentation task: ##############

    # Load CoPick root
    copickRoot = CopickRootFSSpec.from_file(predict_config)

    seg  = Segment(Ncl=n_class, path_weights=path_weights, patch_size=patch_size, gpuID = locGPU)

    # # Load Evaluate TomoIDs
    evalTomos = [run.name for run in copickRoot.runs]

    # Create Temporary Empty Folder 
    for tomoInd in range(len(evalTomos)): 

        if (tomoInd + 1) % nProcess == rank: 

            # Extract TomoID and Associated Run
            tomoID = evalTomos[tomoInd]

            # Load data:
            tomo = tools.get_copick_tomogram(copickRoot, voxelSize=voxel_size, tomoAlgorithm=tomogram_algorithm, tomoID=tomoID)

            # Segment tomogram:
            scoremaps = seg.launch(tomo[:])

            # Get labelmap from scoremaps:
            labelmap  = sm.to_labelmap(scoremaps)
            copickRun = copickRoot.get_run(tomoID)
            tools.write_ome_zarr_segmentation(copickRun, labelmap, voxel_size, user_id, session_id)    

    print('Segmentations Complete!')

if __name__ == "__main__":
    cli()