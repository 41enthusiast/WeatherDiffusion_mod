from gradio_utils import *
import gradio as gr


def cleanup():
    import os, signal
    print("Forcing VRAM cleanup...")
    global model
    # 1. Move model to CPU first (clears dedicated VRAM)
    if 'model' in globals():
        model.to('cpu')
        del model
    
    # 2. Clear Python's garbage collector
    gc.collect()
    
    # 3. Flush the PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("GPU Memory Cleared.")

    # 2. Kill the process itself
    os.kill(os.getpid(), signal.SIGTERM)
    return "Process Terminated"

# -----------------------------
# GRADIO INTERFACE
# -----------------------------

with gr.Blocks() as demo:
    gr.Markdown("# Weather Diffusion Model Demo")
    with gr.Row():
        with gr.Column():
            img_file = gr.File(label="Upload an image file")
        # with gr.Column():
        #     mask_file = gr.File(label="Upload a degradation mask file")
        with gr.Column():
            gt_file = gr.File(label = 'Upload the clean image file (Optional)')
    with gr.Row():
        # with gr.Column():
        #     input_image = gr.Image(label="Input Image")
        # with gr.Column():
        #     mask_image = gr.Image(label="Input Mask")
        with gr.Column():
            masked_image = gr.Image(label="Masked Image")
        with gr.Column():
            gr.Markdown("## Instructions")
            gr.Markdown(
                """
                1. Upload an image with weather degradation (e.g., rain, snow).
                2. Click 'Run Inference' to see the restored image.
                """
            )
            run_button = gr.Button("Run Inference")
    
    status_text = gr.Textbox(label="Status")
    with gr.Row():
        with gr.Column():
            # Component 1: The fast-updating patch grid
            grid_preview = gr.Image(label="Current Patching Progress")
            # Component 2: The step-by-step denoising result
            step_preview = gr.Image(label="Intermediate Denoised Result")
            # Component 3: Final clean output
            final_output = gr.Image(label="Final Restored Image")
    with gr.Row():
        with gr.Column():
            heatmap_image = gr.Image(label="Heatmap difference Image")
        with gr.Column():
            gt_image = gr.Image(label = 'Ground Truth Image')

    with gr.Row():
        panic_btn = gr.Button("Hard Reset & Exit", variant="stop")
    panic_btn.click(fn=cleanup)
    
    img_file.upload(fn = load_image,
                    inputs = img_file,
                    outputs = masked_image)
    gt_file.upload(fn = load_image,
                   inputs = gt_file,
                   outputs = gt_image)

    run_button.click(fn=run_reverse_diffusion,
                     inputs=[masked_image, gt_image],
                     outputs=[status_text, grid_preview, step_preview, final_output, heatmap_image])
try:
    demo.launch()
finally:
    cleanup()