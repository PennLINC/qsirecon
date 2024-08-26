import pandas as pd
from qsirecon.interfaces import recon_scalars
from qsirecon.data import load as load_data
from qsirecon.utils.misc import load_yaml


def scalars_to_csv(scalar_def, output_csv):
    print("creating", output_csv)
    as_data = [
        {
            "Model": value["bids"]["model"],
            "Parameter": value["bids"].get("param", ""),
            "Description": value["metadata"].get("Description", ""),
        }
        for value in scalar_def.values()
    ]
    df = pd.DataFrame(as_data).sort_values(["Model", "Parameter"])
    df.to_csv(output_csv, header=False, index=None)


## TORTOISE.json
scalars_to_csv(recon_scalars.tortoise_scalars, "recon_scalars/tortoise.csv")
## amico_noddi.json
scalars_to_csv(recon_scalars.amico_scalars, "recon_scalars/amico_noddi.csv")
## csdsi_3dshore.json
scalars_to_csv(recon_scalars.brainsuite_3dshore_scalars, "recon_scalars/csdsi_3dshore.csv")
## dipy_3dshore.json
scalars_to_csv(recon_scalars.brainsuite_3dshore_scalars, "recon_scalars/dipy_3dshore.csv")
## dipy_dki.json
scalars_to_csv(recon_scalars.dipy_dki_scalars, "recon_scalars/dipy_dki.csv")
## dipy_mapmri.json
scalars_to_csv(recon_scalars.dipy_mapmri_scalars, "recon_scalars/dipy_mapmri.csv")
## dsi_studio_gqi.json
scalars_to_csv(recon_scalars.dsistudio_scalars, "recon_scalars/dsi_studio_gqi.csv")
## hbcd_scalar_maps.json
scalars_to_csv(
    recon_scalars.amico_scalars
    | recon_scalars.dipy_dki_scalars
    | recon_scalars.tortoise_scalars
    | recon_scalars.dsistudio_scalars,
    "recon_scalars/hbcd_scalar_maps.csv",
)


def output_to_filepattern(bids):
    file_pattern = []
    if "space" in bids:
        file_pattern.append(f"space-{bids['space']}")
    if "atlas" in bids:
        file_pattern.append(f"atlas-{bids['atlas']}")
    if "bundle" in bids:
        file_pattern.append(f"bundle-{bids['bundle']}")
    if "bundles" in bids:
        file_pattern.append(f"bundles-{bids['bundles']}")
    if "model" in bids:
        file_pattern.append(f"model-{bids['model']}")
    if "param" in bids:
        file_pattern.append(f"param-{bids['param']}")
    if "label" in bids:
        file_pattern.append(f"label-{bids['label']}")

    extension = bids.get("extension", "\*")
    file_pattern.append(f"_{bids.get('suffix', 'dwimap')}.{extension}")
    return "\*" + "\*".join(file_pattern)


def outputs_to_csv(output_def, output_csv):
    as_data = [
        {
            "File": output_to_filepattern(value["bids"]),
            "Description": value["metadata"]["Description"].replace("\n", " "),
        }
        for value in output_def.values()
    ]
    df = pd.DataFrame(as_data).sort_values("File")
    df.to_csv(output_csv, header=False, index=None)


outputs_to_csv(load_yaml(load_data("nonscalars/amico_noddi.yaml")), "nonscalars/amico_noddi.csv")
outputs_to_csv(load_yaml(load_data("nonscalars/dipy_dki.yaml")), "nonscalars/dipy_dki.csv")
outputs_to_csv(load_yaml(load_data("nonscalars/dipy_mapmri.yaml")), "nonscalars/dipy_mapmri.csv")
outputs_to_csv(
    load_yaml(load_data("nonscalars/dsistudio_gqi.yaml")), "nonscalars/dsistudio_gqi.csv"
)
outputs_to_csv(
    load_yaml(load_data("nonscalars/dsistudio_autotrack.yaml")),
    "nonscalars/dsistudio_autotrack.csv",
)
outputs_to_csv(load_yaml(load_data("nonscalars/tortoise.yaml")), "nonscalars/tortoise.csv")
outputs_to_csv(
    load_yaml(load_data("nonscalars/csdsi_3dshore.yaml")), "nonscalars/csdsi_3dshore.csv"
)
outputs_to_csv(
    load_yaml(load_data("nonscalars/ss3t_autotrack.yaml")), "nonscalars/ss3t_autotrack.csv"
)
outputs_to_csv(load_yaml(load_data("nonscalars/mrtrix_dwi.yaml")), "nonscalars/mrtrix_dwi.csv")
outputs_to_csv(load_yaml(load_data("nonscalars/mrtrix_anat.yaml")), "nonscalars/mrtrix_anat.csv")
