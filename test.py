def sample_3d_pts_for_pixels(self, pixels, return_depth=False, det=True, near_depth=None, far_depth=None):
        '''
        stratified sampling
        对每个像素采样射线上的点
        :param pixels: [n_imgs, n_pts, 2] 像素值
        :param return_depth: True or False 是否返回深度值
        :param det: if deterministic 是否确定性采样
        :param near_depth: nearest depth 最近深度
        :param far_depth: farthest depth 最远深度
        :return: sampled 3d locations [n_imgs, n_pts, n_samples, 3] 采样得到的三维坐标
        '''
        if near_depth is not None and far_depth is not None:
            assert pixels.shape[:-1] == near_depth.shape[:-1] == far_depth.shape[:-1]
            depths = near_depth + \
                     torch.linspace(0, 1., self.args.num_samples_ray, device=self.device)[None, None] \
                     * (far_depth - near_depth)  # 计算深度范围内的均匀间隔采样点
        else:
            depths = torch.linspace(self.args.min_depth, self.args.max_depth, self.args.num_samples_ray,
                                    device=self.device)  # 在设定的深度范围内均匀采样
            pixels_shape = pixels.shape
            depths = depths[None, None, :].expand(*pixels_shape[:2], -1)  # 扩展维度以匹配像素维度
        if not det:
            # get intervals between samples 获取采样点之间的间隔
            mids = .5 * (depths[..., 1:] + depths[..., :-1])
            upper = torch.cat([mids, depths[..., -1:]], dim=-1)
            lower = torch.cat([depths[..., 0:1], mids], dim=-1)
            # uniform samples in those intervals 在这些间隔内均匀采样
            t_rand = torch.rand_like(depths)
            depths = lower + (upper - lower) * t_rand  # [n_imgs, n_pts, n_samples]

        depths = depths[..., None]
        pixels_expand = pixels.unsqueeze(-2).expand(-1, -1, self.args.num_samples_ray, -1)

        x = self.unproject(pixels_expand, depths)  # [n_imgs, n_pts, n_samples, 3] 三维采样点坐标
        if return_depth:
            return x, depths
        else:
            return x
