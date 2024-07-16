import os
import ffmpeg
import subprocess

def convert_m4a_to_wav(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 指定 ffmpeg 可执行文件的路径
    ffmpeg_path = 'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.m4a'):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + '.wav')

            # 使用 ffmpeg 进行转换
            try:
                process = (
                    ffmpeg
                    .input(input_file)
                    .output(output_file, acodec='pcm_s16le', ar='44100')
                    .run_async(cmd=ffmpeg_path, pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
                )
                process.wait()
                print(f'Converted {input_file} to {output_file}')
            except ffmpeg.Error as e:
                print(f'Error converting {input_file}: {e}')

# 示例用法
input_folder = 'c:/Users/herrw/Desktop/NGM'
output_folder = 'c:/Users/herrw/Desktop/NG'
convert_m4a_to_wav(input_folder, output_folder)