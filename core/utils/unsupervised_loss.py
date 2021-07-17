import torch
import torch.nn as nn
import torch.nn.functional as F

# adapted from "Back to Basics: Unsupervised Learning of Optical Flow via Brightness"
# Constancy and Motion Smoothness

def charbonnier(x, alpha=0.25, epsilon=1.e-9):
	return torch.pow(torch.pow(x, 2) + epsilon**2, alpha)

def smoothness_loss(flow):
	b, c, h, w = flow.size()
	v_translated = torch.cat((flow[:, :, 1:, :], torch.zeros(b, c, 1, w, device=flow.device)), dim=-2)
	h_translated = torch.cat((flow[:, :, :, 1:], torch.zeros(b, c, h, 1, device=flow.device)), dim=-1)
	s_loss = charbonnier(flow - v_translated) + charbonnier(flow - h_translated)
	s_loss = torch.sum(s_loss, dim=1) / 2

	return torch.sum(s_loss)/b

def photometric_loss(wraped, frame1):	
	h, w = wraped.shape[2:]
	frame1 = F.interpolate(frame1, (h, w), mode='bilinear', align_corners=False)
	p_loss = charbonnier(wraped - frame1)
	p_loss = torch.sum(p_loss, dim=1)/3
	return torch.sum(p_loss)/frame1.size(0)

def unsup_loss(pred_flows, warped_images, frame1):
	bce =  photometric_loss(warped_images, frame1)
	smooth = smoothness_loss(pred_flows)
	loss = bce + smooth
	return loss