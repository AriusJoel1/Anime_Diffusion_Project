@echo off
python generate.py --checkpoint checkpoints/best.pt --num-images 16 --sampling-steps 100
pause
