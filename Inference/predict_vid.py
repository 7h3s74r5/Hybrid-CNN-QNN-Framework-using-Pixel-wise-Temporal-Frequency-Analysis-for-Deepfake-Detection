CLIP_LENGTH = 32


def predict_video(

    video_path,

    model,

    device

):

    predictions = []


    # -----------------------------
    # Extract faces
    # -----------------------------

    frames = extract_faces(

        video_path

    )


    if len(frames) < CLIP_LENGTH:

        return None


    # -----------------------------
    # Create clips
    # -----------------------------

    for i in range(

        0,

        len(frames)-CLIP_LENGTH+1,

        CLIP_LENGTH

    ):


        clip = frames[

            i:i+CLIP_LENGTH

        ]


        # -------------------------
        # PwTF
        # -------------------------

        pwtf = generate_pwtf(

            clip

        )


        # -------------------------
        # Model Prediction
        # -------------------------

        prob = predict_pwtf_clip(

            pwtf,

            model,

            device

        )


        predictions.append(

            prob

        )


    if len(predictions)==0:

        return None


    # Average all clips

    final_probability = np.mean(

        predictions

    )


    if final_probability >= 0.5:

        label="Fake"

    else:

        label="Real"



    confidence = (

        final_probability

        if label=="Fake"

        else

        1-final_probability

    )


    return {

        "prediction":label,

        "confidence":confidence,

        "fake_probability":final_probability,

        "num_clips":len(predictions)

    }
