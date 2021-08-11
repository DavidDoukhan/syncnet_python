# to truncate a video
# ffmpeg -ss 00:00:00.0 -i kMy-6RtoOVU.mkv -c copy -t 00:06:49.0 kMy-6RtoOVU.tru3.mkv


f = open('../AVA_talknet/csv/val_orig.csv', 'r')
f2 = open('../AVA_talknet_lite/csv/val_orig.csv', 'w')

for row in f:
    if (row.startswith('kMy-6RtoOVU')):
        f2.write(row)