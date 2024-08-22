import pandas as pd
from qsirecon.interfaces import recon_scalars


def scalars_to_csv(scalar_def, output_csv):
    as_data = [
        {
            "Model": value["bids"]["model"],
            "Parameter": value["bids"]["param"],
            "Description": value["desc"],
        }
        for value in scalar_def.values()
    ]
    df = pd.DataFrame(as_data).sort_values(["Model", "Parameter"])
    df.to_csv(output_csv, header=False, index=None)


## TORTOISE.yaml
scalars_to_csv(recon_scalars.tortoise_scalars, "recon_scalars/tortoise.csv")
## amico_noddi.yaml
scalars_to_csv(recon_scalars.amico_scalars, "recon_scalars/amico_noddi.csv")
## csdsi_3dshore.yaml
scalars_to_csv(recon_scalars.brainsuite_3dshore_scalars, "recon_scalars/csdsi_3dshore.csv")
## dipy_3dshore.yaml
scalars_to_csv(recon_scalars.brainsuite_3dshore_scalars, "recon_scalars/dipy_3dshore.csv")
## dipy_dki.yaml
scalars_to_csv(recon_scalars.dipy_dki_scalars, "recon_scalars/dipy_dki.csv")
## dipy_mapmri.yaml
scalars_to_csv(recon_scalars.dipy_mapmri_scalars, "recon_scalars/dipy_mapmri.csv")
## dsi_studio_gqi.yaml
scalars_to_csv(recon_scalars.dsistudio_scalars, "recon_scalars/dsi_studio_gqi.csv")
## hbcd_scalar_maps.yaml
scalars_to_csv(
    recon_scalars.amico_scalars
    | recon_scalars.dipy_dki_scalars
    | recon_scalars.tortoise_scalars
    | recon_scalars.dsistudio_scalars,
    "recon_scalars/hbcd_scalar_maps.csv",
)
