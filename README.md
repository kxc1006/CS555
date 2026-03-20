# Pipette Procedure Understanding Notes

This README records my current implementation, findings, and proposed next direction for the pipetting video reasoning task.

## 1. Current baseline: SDLS

SDLS repo:  
https://github.com/letsgo-2/sdls-abnormaldetect/tree/master

Script location:  
`sdls/scripts`

Current finding:

SDLS is essentially a single-frame prompting-based abnormal detection method. It can detect simple static actions such as whether an insert-like state is present, but it suffers from the tape-trap problem when applied to pipetting videos.

In particular, SDLS depends heavily on prompt wording and peak-score matching, and it cannot reliably reason over continuous temporal logic such as:

`Approaching -> Inserted -> Injecting -> Withdrawn -> Leaving`

Therefore, SDLS may serve as a weak local event baseline, but it is not sufficient for full pipetting procedure understanding.

## 2. Core action states and abnormal behavior definitions

### Normal temporal procedure

The pipetting process is decomposed into the following symbolic states:

1. Approaching  
   The pipette moves toward the target well region.

2. Inserted  
   The pipette tip is inserted into the designated well.

3. Injecting  
   Liquid is dispensed, which should correspond to a continuous decrease of visible liquid height or liquid volume inside the tip.

4. Withdrawn  
   The pipette tip is pulled out of the well.

5. Leaving  
   The pipette leaves the plate region.

### Abnormal behavior targets

1. Trajectory Error  
   The pipette inserts into the wrong well, follows an incorrect spatial path, or violates the expected temporal order.

2. Empty Injection  
   An insertion action is detected, but no actual liquid reduction is observed inside the pipette tip.

3. Liquid Spillage  
   Liquid leakage, overflow, or unintended droplet release is observed outside the target well.

## 3. Current direction

Instead of asking a VLM to directly interpret raw pipetting videos, the current direction is to introduce lightweight spatial anchors and visual state extraction first, then perform symbolic reasoning on top of the extracted states.

The main idea is:

Step 1  
Use a lightweight visual detector to extract object locations, relative spatial relations, and visible liquid-height changes.

Step 2  
Convert frame-level detections into symbolic states such as Approaching, Inserted, Injecting, Withdrawn, and Leaving.

Step 3  
Feed the symbolic state sequence into a VLM or LLM for higher-level reasoning about whether the procedure is correct or abnormal.

This design avoids forcing the VLM to directly parse noisy raw video frames. Instead, the reasoning model operates on structured visual evidence, which should improve robustness, interpretability, and temporal reasoning quality.

## 4. Lightweight visual anchors

The current visual detector is designed to track:

- `liquid_tip`
- `pipette`
- `plate`

The detector is expected to extract observable visual variables such as:

- plate region position
- pipette position
- liquid_tip position
- relative relation between liquid_tip and plate
- visible liquid-height change inside the liquid_tip

Example detector output:
<img width="1194" height="667" alt="image" src="https://github.com/user-attachments/assets/ade8f709-b2fa-4d52-bdb5-a80f7bfbf115" />

```json
{
  "predictions": [
    {
      "x": 751.5,
      "y": 675,
      "width": 1077,
      "height": 196,
      "confidence": 0.981,
      "class": "plate",
      "class_id": 2,
      "detection_id": "93387666-58e7-4563-ba45-bf0764154521"
    },
    {
      "x": 748.5,
      "y": 597,
      "width": 115,
      "height": 188,
      "confidence": 0.978,
      "class": "liquid_tip",
      "class_id": 0,
      "detection_id": "8411a0a6-d27f-4a14-8e12-c451da3dc10a"
    },
    {
      "x": 745,
      "y": 432.5,
      "width": 128,
      "height": 515,
      "confidence": 0.969,
      "class": "pipette",
      "class_id": 1,
      "detection_id": "cc258337-292f-42cd-adbb-283f5453464a"
    }
  ]
}



