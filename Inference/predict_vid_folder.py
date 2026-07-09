VIDEO_FOLDER="/content/test_videos"


results=[]


videos=os.listdir(

    VIDEO_FOLDER

)


for video in tqdm(videos):


    if not video.lower().endswith(

        (".mp4",".avi",".mov",".mkv")

    ):

        continue



    path=os.path.join(

        VIDEO_FOLDER,

        video

    )


    result=predict_video(

        path,

        model,

        device

    )


    if result is not None:


        results.append(

            {

            "video":video,

            **result

            }

        )

