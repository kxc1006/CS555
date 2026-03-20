import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from pathlib import Path
import av  
import argparse
from scipy.signal import find_peaks 

SAMPLE_RATE = 15  

#TEXT_PROMPTS = { 
#    "State: Hovering": "A macro photo of empty 96-well microplate wells, no pipette tips inside",         
#    "State: Inserted": "Macro photo of plastic pipette tips fully inserted into the wells of a microplate"
#}

TEXT_PROMPTS = { 
    #"State: Hovering": "A macro photo of empty 96-well microplate wells",         
    "State: Inserted": "Macro photo of plastic pipette tips fully inserted into the wells of a microplate"
}

def load_model():
    print("CLIP Loading...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor, device

def process_video(video_path, model, processor, device, output_dir):
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
    except Exception as e:
        print(f"can't open video: {video_path}, error: {e}")
        return None, None

    fps = float(stream.average_rate)
    
    text_labels = list(TEXT_PROMPTS.keys())
    text_sentences = list(TEXT_PROMPTS.values())
    
    results = {label: [] for label in text_labels}
    time_points = []
    
    print("evaluating (Bottom ROI Crop + Spatial State Logic)...")
    frame_idx = 0
    saved_debug = False 

    for frame in container.decode(video=0):
        if frame_idx % SAMPLE_RATE == 0:
            pil_image = frame.to_image()
            w, h = pil_image.size
            
            top_crop = int(h * 0.45)
            roi_image = pil_image.crop((0, top_crop, w, h))

            if not saved_debug:
                debug_path = output_dir / "debug_roi_view.jpg" 
                roi_image.save(debug_path)
                print(f"Debug: {debug_path}")
                saved_debug = True

            inputs = processor(
                text=text_sentences, 
                images=roi_image, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
                text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

                similarity = image_features @ text_features.T
                scores = similarity[0].cpu().numpy()
                
                current_time = frame_idx / fps
                time_points.append(current_time)
                
                for i, label in enumerate(text_labels):
                    results[label].append(scores[i])

        frame_idx += 1

    container.close()
    return time_points, results

def plot_results(time_points, results, output_dir):
    plt.figure(figsize=(12, 6))
    colors = {"State: Hovering": "#1f77b4", "State: Inserted": "#ff7f0e"}
    
    for label, scores in results.items():
        plt.plot(time_points, scores, label=label, color=colors[label], linewidth=2.5, alpha=0.8)

    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("CLIP Similarity Score", fontsize=12)
    plt.title("Action Sequence (ROI Spatial Event Detection)", fontsize=14)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_path = output_dir / "step_detection_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nfigure saved to: {plot_path}")

def detect_anomalies(time_points, results):
    print("\n" + "="*40)
    print("evaluating process...")
    print("="*40)
    
    insert_scores = results["State: Inserted"]
    
    window_size = 3 
    if len(insert_scores) >= window_size:
        smoothed_insert = np.convolve(insert_scores, np.ones(window_size)/window_size, mode='same')
    else:
        smoothed_insert = np.array(insert_scores)
        
    baseline_insert = np.percentile(smoothed_insert, 25)
    
    REQUIRED_PROMINENCE = 0.035
    ABSOLUTE_MIN_INSERTION = 0.285 
    
    min_distance_frames = max(1, int(2.0 / (time_points[1] - time_points[0])))

    peaks, properties = find_peaks(
        smoothed_insert,
        height=ABSOLUTE_MIN_INSERTION,
        prominence=REQUIRED_PROMINENCE,
        distance=min_distance_frames
    )
    
    print(f" (Baseline): {baseline_insert:.3f}")
    
    if len(peaks) == 0:
        max_insert_score = np.max(smoothed_insert)
        prominence = max_insert_score - baseline_insert
        print(f"(Peak): {max_insert_score:.3f}")
        print(f"(Prominence): {prominence:.3f}")
        
        if max_insert_score < ABSOLUTE_MIN_INSERTION:
            print(f"\n ERROR: lack of movement")
            print(f"   max_insert_score({max_insert_score:.3f}) not achieve ABSOLUTE_MIN_INSERTION({ABSOLUTE_MIN_INSERTION})。")
            print("   evaluate: no reall touch")
        else:
            print(f"\n ERROR: lack of movement depth")
            print(f"   reason: prominence ({prominence:.3f}) submerged by backgrounf noise ({REQUIRED_PROMINENCE})。")
        print("="*40 + "\n")
        return

    print("\n[process decicsion]:")
    for i, peak_idx in enumerate(peaks):
        peak_time = time_points[peak_idx]
        peak_score = smoothed_insert[peak_idx]
        prom_val = properties['prominences'][i]
        
        print(f"\n--- Action {i+1} ---")
        print(f"(Peak): {peak_score:.3f} (peak time:{peak_time:.1f}s)")
        print(f"(Prominence): {prom_val:.3f}")
        print(f"(Dispensing)，appeared on: {peak_time:.1f}second")
        print(f"real movement: Pipette Approaching -> Dispensing -> Pipette Withdrawing")

    print(f"\n PASS (Total valid actions: {len(peaks)})")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDLS Final - Hybrid Guardrails Method")
    parser.add_argument("video_name", type=str, help="Video file name")
    args = parser.parse_args()

    video_filename = Path(args.video_name).name 
    video_stem = Path(video_filename).stem
    
    input_video_path = Path("videos") / video_filename
    output_dir = Path("videos_output") / video_stem
                          
    if not input_video_path.exists():
        print(f"error: no file named with '{input_video_path}'")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

        model, processor, device = load_model()
        times, scores = process_video(input_video_path, model, processor, device, output_dir)
        
        if times:
            plot_results(times, scores, output_dir)
            detect_anomalies(times, scores)