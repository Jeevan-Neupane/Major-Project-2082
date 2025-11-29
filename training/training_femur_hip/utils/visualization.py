import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_swinx2s_plotly(volume, threshold=0.5, colorscale="Gray"):
    vol = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-8)
    D, H, W = vol.shape
    
    x = np.arange(D).repeat(H * W)
    y = np.tile(np.arange(H).repeat(W), D)
    z = np.tile(np.arange(W), D * H)
    vol_flat = vol.flatten()

    fig = go.Figure(data=go.Isosurface(
        x=x, y=y, z=z, value=vol_flat,
        isomin=threshold, isomax=1.0, surface_count=1,
        colorscale=colorscale,
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.7
    ))

    fig.update_layout(
        title="SwinX2S 3D Reconstruction",
        scene=dict(xaxis_title="Depth (D)", yaxis_title="Height (H)", zaxis_title="Width (W)", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.show()

def render_ct_plotly_resampled(nii_path, target_shape=(64, 128, 128), threshold=0.5):
    vol = nib.load(nii_path).get_fdata().astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    
    vol_t = torch.tensor(vol)[None, None, ...]
    vol_resampled = F.interpolate(vol_t, size=target_shape, mode='trilinear', align_corners=False)
    vol_resampled = vol_resampled[0, 0].cpu().numpy()
    
    D, H, W = vol_resampled.shape
    x = np.arange(D).repeat(H * W)
    y = np.tile(np.arange(H).repeat(W), D)
    z = np.tile(np.arange(W), D * H)
    vol_flat = vol_resampled.flatten()

    fig = go.Figure(data=go.Isosurface(
        x=x, y=y, z=z, value=vol_flat,
        isomin=threshold, isomax=1.0, surface_count=1,
        colorscale="Gray",
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.7
    ))

    fig.update_layout(
        title=f"CT Volume (Resampled to {target_shape})",
        scene=dict(xaxis_title="Depth (D)", yaxis_title="Height (H)", zaxis_title="Width (W)", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.show()

def render_pred_vs_ct_side_by_side(pred_volume, ct_nii_path, threshold_pred=0.5, threshold_ct=0.5,
                                    colorscale_pred="Viridis", colorscale_ct="Gray"):
    pred = np.asarray(pred_volume, dtype=np.float32)
    D, H, W = pred.shape
    target_shape = (D, H, W)

    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    ct = nib.load(ct_nii_path).get_fdata().astype(np.float32)
    ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-8)

    ct_t = torch.from_numpy(ct)[None, None, ...]
    ct_rs = F.interpolate(ct_t, size=target_shape, mode="trilinear", align_corners=False)
    ct_rs = ct_rs[0, 0].cpu().numpy()

    def _coords_for(vol):
        d, h, w = vol.shape
        x = np.arange(d).repeat(h * w)
        y = np.tile(np.arange(h).repeat(w), d)
        z = np.tile(np.arange(w), d * h)
        return x, y, z, vol.flatten()

    x_p, y_p, z_p, v_p = _coords_for(pred)
    x_c, y_c, z_c, v_c = _coords_for(ct_rs)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type":"scene"}, {"type":"scene"}]],
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.04,
        subplot_titles=("Prediction (SwinX2S)", "Real CT (resampled)")
    )

    fig.add_trace(go.Isosurface(
        x=x_p, y=y_p, z=z_p, value=v_p,
        isomin=threshold_pred, isomax=1.0, surface_count=1,
        colorscale=colorscale_pred,
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.8, name="Prediction"
    ), row=1, col=1)

    fig.add_trace(go.Isosurface(
        x=x_c, y=y_c, z=z_c, value=v_c,
        isomin=threshold_ct, isomax=1.0, surface_count=1,
        colorscale=colorscale_ct,
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.7, name="CT"
    ), row=1, col=2)

    fig.update_layout(
        title=f"SwinX2S Prediction vs Real CT (both {target_shape})",
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(xaxis_title="Depth (D)", yaxis_title="Height (H)", zaxis_title="Width (W)", aspectmode="data"),
        scene2=dict(xaxis_title="Depth (D)", yaxis_title="Height (H)", zaxis_title="Width (W)", aspectmode="data"),
        showlegend=False
    )
    fig.show()