REUSABLE VIDEO BACKGROUND REPLACER

1. Put the raw presenter video and one background image in the input folder.
2. Double-click RUN_BACKGROUND_REPLACER.bat.
3. If there is only one video and one image, they are selected automatically.
   If there are several, choose each one by number when asked.
4. Find the completed video in the output folder. Its name is automatically
   based on the raw video's name and ends with _NEW_BACKGROUND.mp4.

For a 30-second preview, double-click RUN_30_SECOND_PREVIEW.bat. It uses the
same automatically selected input video and background image and creates a
separate preview file.

You do not need to edit the Python script. To use another video later, remove
or move the old input files, place the new raw video and background image in
input, and run the BAT file again.

For a video kept elsewhere, run from PowerShell with absolute paths:
  $env:BACKGROUND_VIDEO_PATH="D:\\MVI_1kjednkckj352.MP4"
  $env:BACKGROUND_IMAGE_PATH="D:\\missionasset\\background_replacer\\input\\ChatGPT Image Sep 1, 2026, 10_05_04 PM.png"
  python D:\\missionasset\\background_replacer\\replace_background.py

The script:
- detects and removes the original human-video background;
- uses the memory-safe human-segmentation mask and edge cleanup to reduce
  green/white halos around the person's hair, shoulders, and clothing;
- keeps the matte frame-accurate during fast head and hand movement;
- adds a subtle contact shadow without an artificial light halo;
- matches the subject tone and adds shared fine grain so the composite feels
  like one camera shot;
- composites the cleaned subject over the supplied image;
- fits the background to 1920x1080 without stretching;
- applies a controlled background blur;
- keeps the original video audio, if present.
- stores temporary extracted frames on the same D: drive inside temp/;
  the Windows C: drive is not used for the large frame cache.

The default balanced mode analyzes at 12 FPS at 768x432 using the human-specific
u2net_human_seg model, then exports the final video at 30 FPS. It is faster than
the previous high-resolution mode while retaining the stronger cleanup for
hair, face, shoulder, ear, and moving-hand edges. A much larger model was tested
but exceeded the laptop's available memory.
For extremely fast movement, increase PROCESSING_FPS and SEGMENTATION_SIZE in
the Python file, but this will increase render time. If the person is too large
or small, adjust SUBJECT_SCALE. The first run may download the rembg model
automatically. If Windows becomes memory-limited, keep WORKERS at 2.
