import cv2
import numpy as np 
import sys
import subprocess as sp

def Streaming(frame):
        
    FFMPEG_BIN = "ffmpeg" # on Linux ans Mac OS

    video_path= 0#sys.argv[1]
    youtube_live_key = 'qm6e-e2f7-7kuk-r2qh-eebd'

    command = [ FFMPEG_BIN,
        '-i', '-', #input from pipe
        '-f', 'lavfi',
        '-i','anullsrc', #Youtube Live needs some audio, so create fake audio src
        '-acodec', 'libmp3lame', 
        '-ar', '44100',
        '-deinterlace', 
        '-vcodec', 'libx264', #Use libx264 if no nvidia GPU available
        '-pix_fmt', 'yuv420p',
        '-s', '1280x720',
        '-preset', 'ultrafast',
        '-tune', 'fastdecode',
        '-r', '30', #Frame rate
        '-g', '120', 
        '-b:v', '2500k',
        '-threads:6', 
        '-qscale:3',
        '-b:a:712000',
        '-buffsize:512k',
        '-f' , 'flv',
        'rtmp://a.rtmp.youtube.com/live2/{}'.format(youtube_live_key),
        ]

    pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)
    ret = True
    if ret:
        ret2,frame_out = cv2.imencode('.jpg',frame) #Breaks without this for some reason
        if ret2:
            pipe.stdin.write( frame_out.tobytes() )
