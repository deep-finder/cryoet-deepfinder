from deepfinder.training_copick import Train
import deepfinder.utils.common as cm
from typing import List, Tuple
import copick, click, os

@click.group()
@click.pass_context
def cli(ctx):
    pass


@cli.command()
@click.option(
    "--path-train",
    type=str,
    required=True,
    help="Path to the copick config file (if --path-valid is provided as well, this is the training split).",
)
@click.option(
    "--path-valid",
    type=str,
    required=False,
    default=None,
    show_default=True,
    help="Path to the copick config file (if provided, this is the validation split).",
)
@click.option(
    "--train-voxel-size",
    type=float,
    required=True,
    help="Voxel size of the tomograms.",
)
@click.option(
    "--train-tomo-type",
    type=str,
    required=True,
    help="Type of tomograms used for training.",
)
@click.option(
    "--target",
    type=(str, str, str),
    required=False,
    help="Tuples of object name, user id, session id and radius.",
    multiple=True,
)
@click.option(
    "--output-path",
    type=str,
    required=True,
    help="Path to store the training results.",
)
@click.option(
    "--n-class",
    type=int,
    required=True,
    help="Number of classes.",
)
@click.option(
    "--dim-in",
    type=int,
    required=False,
    default=52,
    show_default=True,
    help="Patch size.",
)
@click.option(
    "--n-sub-epoch",
    type=int,
    required=False,
    default=10,
    show_default=True,
    help="Number of epochs to train on a subset of the data.",
)
@click.option(
    "--sample-size",
    type=int,
    required=False,
    default=15,
    show_default=True,
    help="Size of the subset of tomos to load into memory.",
)
@click.option(
    "--batch-size",
    type=int,
    required=False,
    default=15,
    show_default=True,
    help="Batch size.",
)
@click.option(
    "--epochs",
    type=int,
    required=False,
    default=65,
    show_default=True,
    help="Number of epochs.",
)
@click.option(
    "--steps-per-epoch",
    type=int,
    required=False,
    default=250,
    show_default=True,
    help="Number of steps per epoch.",
)
@click.option(
    "--n-valid",
    type=int,
    required=False,
    default=20,
    show_default=True,
    help="Number of steps per validation.",
)
@click.option(
    "--lrnd",
    type=int,
    required=False,
    default=13,
    show_default=True,
    help="Random shifts when sampling patches (data augmentation).",
)
@click.option(
    "--direct-read/--no-direct-read",
    type=bool,
    is_flag=True,
    required=False,
    default=False,
    show_default=True,
    help="Flag to directly read the data.",
)
@click.option(
    "--batch-bootstrap/--no-batch-bootstrap",
    type=bool,
    is_flag=True,
    required=False,
    default=True,
    show_default=True,
    help="Flag to bootstrap the batches.",
)
@click.option(
    "--label-name",
    type=str,
    required=False,
    default="spheretargets",
    show_default=True,
    help="Name of the segmentation target.",
)
@click.option(
    "--label-user-id",
    type=str,
    required=False,
    default="train-deepfinder",
    show_default=True,
    help="User ID of the segmentation target.",
)
# @click.option(
#     "--class-weight",
#     type=(str, float),
#     multiple=True,
#     required=False,
#     default=None,
#     show_default=True,
#     help="Class weights.",
# )
@click.option(
    "--valid-tomo-ids",
    type=str,
    required=False,
    default=None,
    show_default=True,
    help="List of validation tomoIDs.",
)
@click.option(
    "--train-tomo-ids",
    type=str,
    required=False,
    default=None,
    show_default=True,
    help="List of training tomoIDs.",
)
def train(
    path_train: str,
    train_voxel_size: int,
    train_tomo_type: str,
    target: List[Tuple[str, str, str]],
    output_path: str,
    n_class: int,
    path_valid: str = None,
    dim_in: int = 52,
    n_sub_epoch: int = 10,
    sample_size: int = 15,
    batch_size: int = 15,
    epochs: int = 65,
    steps_per_epoch: int = 250,
    n_valid: int = 20,
    lrnd: int = 13,
    direct_read: bool = False,
    batch_bootstrap: bool = True,
    label_name: str = "spheretargets",
    label_user_id: str = "train-deepfinder",
    # class_weight: dict = None,
    valid_tomo_ids: str = None,
    train_tomo_ids: str = None,
):
    # Parse input parameters
    if valid_tomo_ids is not None:
        valid_tomo_ids = valid_tomo_ids.split(",")
    if train_tomo_ids is not None:
        train_tomo_ids = train_tomo_ids.split(",")

    # Copick Input Parameters
    trainVoxelSize = train_voxel_size
    trainTomoAlg = train_tomo_type

    # Input parameters:
    Nclass = n_class

    # Initialize training task:
    trainer = Train(Ncl=Nclass, dim_in=dim_in)
    trainer.path_out = output_path  # output path
    trainer.batch_size = batch_size
    trainer.epochs = epochs
    trainer.steps_per_epoch = steps_per_epoch
    trainer.Nvalid = n_valid  # steps per validation
    trainer.Lrnd = lrnd  # random shifts when sampling patches (data augmentation)
    trainer.flag_direct_read = direct_read
    trainer.flag_batch_bootstrap = batch_bootstrap

    # Segmentation Target Name And Corresponding UserID
    trainer.labelName = label_name
    trainer.labelUserID = label_user_id

    # Experimental Weights - [background, membrane, apo, betaAmylase, betaGal, ribo80S, thg, vlp]
    trainer.class_weights = None
    # trainer.class_weights    = {0:1, 1:3000, 2:6500, 3:70790, 4:800, 5:20225, 6:10300, 7:28000}

    # Use following line if you want to resume a previous training session:
    # trainer.net.load_weights('synthetic_dataset_10A/training_results/net_weights_FINAL.h5')

    # A Certain Number of Tomograms are Loaded Prior to Training (sample_size)
    # And picks from these tomograms are trained for a specified number of epochs (NsubEpoch)
    trainer.NsubEpoch = n_sub_epoch
    trainer.sample_size = sample_size

    targets = {}
    for t in target:
        info = {
            "user_id": t[1],
            "session_id": t[2],
        }
        targets[t[0]] = info

    trainer.targets = targets

    # Create output Path
    os.makedirs(output_path, exist_ok=True)

    # Copick Input Parameters
    trainer.voxelSize = trainVoxelSize
    trainer.tomoAlg = trainTomoAlg

    # Finally, launch the training procedure:
    if path_valid is None and valid_tomo_ids is None and train_tomo_ids is None:
        # Option 1:
        # Split the Entire Copick Project into Train / Validation / Test
        tomo_ids = [r.name for r in copick.from_file(path_train).runs]
        (trainList, validationList, testList) = cm.split_datasets(
            tomo_ids,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            savePath=output_path,
        )

        # Pass the Run IDs to the Training Class
        trainer.validTomoIDs = validationList
        trainer.trainTomoIDs = trainList

        trainer.launch(path_train)

    elif path_valid is None and valid_tomo_ids is not None and train_tomo_ids is not None:
        # Option 2:
        # train and valid tomoIDs are provided
        trainer.trainTomoIDs = train_tomo_ids
        trainer.validTomoIDs = valid_tomo_ids

        trainer.launch(path_train)

    elif path_valid is not None:
        # Option 3:
        # The Data is Already Split into two Copick Projects

        # Train
        trainer.launch(path_train, path_valid)


if __name__ == "__main__":
    cli()
