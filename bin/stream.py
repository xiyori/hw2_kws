import logging
import argparse
import multiprocessing as mp

import torch
from torchaudio.io import StreamReader

logger = logging.getLogger(__file__)


def audio_stream(queue: mp.Queue, device: str,
                 format: str, frames_per_chunk: int):
    """
    Learn more about how to install and use streaming audio here
    https://pytorch.org/audio/stable/tutorials/streaming_api2_tutorial.html
    """

    streamer = StreamReader(src=device, format=format)
    streamer.add_basic_audio_stream(frames_per_chunk=frames_per_chunk,
                                    buffer_chunk_size=5,
                                    sample_rate=16000)
    stream_iterator = streamer.stream(-1, 1)

    logger.info("Start audio streaming")
    try:
        while True:
            (chunk_,) = next(stream_iterator)
            logger.info("Put chunk to queue")
            queue.put(chunk_)
    except StopIteration:
        queue.put(None)
        return


def parse_args():
    parser = argparse.ArgumentParser(description="Run streaming KWS.")
    parser.add_argument("path", nargs="?", metavar="PATH", type=str,
                        default="resources/checkpoints/streaming_kws.pth",
                        help="Path to streaming KWS model save (default: %(default)s).")
    parser.add_argument("-d", "--device", metavar="DEVICE", type=str,
                        default=r"audio=@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}"
                                r"\wave_{2BE75018-83E5-4BE8-9A76-2D54B49893FD}",
                        help="Input audio device (default: %(default)s).")
    parser.add_argument("-f", "--format", metavar="FORMAT", type=str, default="dshow",
                        help="Input audio format (default: %(default)s).")
    parser.add_argument("-c", "--frames_per_chunk", metavar="CHUNK_SIZE", type=int, default=3840,
                        help="Chunks size in frames (default: %(default)s).")
    parser.add_argument("-w", "--max_window_length", metavar="WINDOW_SIZE", type=int, default=5,
                        help="Sliding window size in chunks (default: %(default)s).")
    parser.add_argument("-r", "--raw", dest="raw", action="store_true",
                        help="Output raw confidence values instead of detection message.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = torch.jit.load(args.path).eval()

    ctx = mp.get_context("spawn")
    chunk_queue = ctx.Queue()
    streaming_process = ctx.Process(target=audio_stream,
                                    args=(chunk_queue, args.device,
                                          args.format, args.frames_per_chunk))

    streaming_process.start()
    accum = torch.tensor([], dtype=torch.float32)
    current_length = 0
    while True:
        try:
            chunk = chunk_queue.get()
            if chunk is None:
                break

            if chunk.dim() == 2:
                chunk = chunk.mean(dim=1)

            if current_length < args.max_window_length:
                current_length += 1
            else:
                step_size = chunk.shape[0]
                accum = accum[step_size:]
            accum = torch.cat([accum, chunk], dim=0)

            # Pad the last chunk
            if accum.shape[0] < args.frames_per_chunk:
                padded_chunk = torch.zeros(args.frames_per_chunk, dtype=accum.dtype)
                padded_chunk[:accum.shape[0]] = accum
                accum = padded_chunk
            # print(f"{chunk.shape}")

            with torch.inference_mode():
                result = model(accum, 1)  # args.max_window_length)

            if args.raw:
                print(result.item())
            elif result > 0.7:
                print("DETECTED KEY WORD")

        except (KeyboardInterrupt, StopIteration):
            break
        except Exception as exc:
            raise exc

    streaming_process.join()
