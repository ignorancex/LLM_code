import torch

class SE3:
    def __init__(self, R=None, t=None, mat=None, se3=None) -> None:
        if mat is not None:
            if mat.shape[-2] == 4:
                self._mat = mat
            else:
                zeros = torch.tensor([0, 0, 0, 1], dtype=mat.dtype, device=mat.device)
                if mat.dim() == 3:
                    self._mat = torch.cat((mat,zeros[None].expand(mat.shape[0],-1, -1)), dim=1)
                else:
                    self._mat = torch.row_stack((mat, zeros))
        elif R is not None:  
            if t is None:
                self._mat = R
            else:  
                self._mat = torch.eye(4, 4, dtype=R.dtype, device=R.device)
                if R.dim() == 3:
                    self._mat = self._mat[None].expand(R.shape[0], 4, 4)    
                self._mat[..., :3, :3] = R
                self._mat[..., :3, 3] = t 
        elif se3 is not None:
            zeros = torch.tensor([0, 0, 0, 1], dtype=se3.dtype, device=se3.device)
            self._mat = torch.row_stack((Lie.se3_to_SE3(se3), zeros))
        self.mat = self._mat
        self.R = self._mat[..., :3, :3]
        self.t = None if (R is not None and t is None) else self._mat[..., :3, 3]

    def __matmul__(self, other):
 
        if isinstance(other, SE3):
            return SE3(mat=self._mat @ other._mat)
        elif isinstance(other, torch.Tensor):
            R, t = self.R, self.t
            if self._mat.dim()==2:
                if t is not None:
                    return (other @ R.transpose(0, 1) + t[None, :])
                else:
                    return other @ R.transpose(0, 1)
            else:
                if t is not None:
                    return torch.bmm(R, other[..., None])[..., 0] + t
                else:
                    return torch.bmm(R, other[..., None])[..., 0]
    
    def __getitem__(self, index):
        if self._mat.dim()<3:
            return self._mat[index]
        return SE3(mat=self._mat[index])
    
    def invert(self):
        inv = torch.linalg.inv(self._mat)
        if self.t is None:
            return SE3(R=inv)
        else:
            return SE3(mat=inv)
    
    @property
    def get_Rt(self):
        return SE3(R=self.R), self.t

    @property
    def get_R(self):
        return SE3(R=self.R)
    
    @property
    def get_t(self):
        return self.t
    
    def se3(self):
        return Lie.SE3_to_se3(self._mat[..., :3, :])

class Camera:
    @staticmethod
    def pixel_to_ray(width, height, intr, mask):
        v, u = torch.meshgrid(torch.arange(height, dtype=torch.float).cuda(), torch.arange(width, dtype=torch.float).cuda())
        uv_ = torch.stack((u, v, torch.ones_like(u)), dim=-1)[mask]
        xyz = intr.invert() @ uv_
        return xyz


class Lie:
    """
    MIT License

    Copyright (c) 2021 Chen-Hsuan Lin

    Reference https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/main/LICENSE
    """
    @staticmethod
    def so3_to_SO3(w): # [...,3]
        wx = Lie.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = Lie.taylor_A(theta)
        B = Lie.taylor_B(theta)
        R = I+A*wx+B*wx@wx
        return R

    @staticmethod
    def SO3_to_so3(R,eps=1e-7): # [...,3,3]
        trace = R[...,0,0]+R[...,1,1]+R[...,2,2]
        theta = ((trace-1)/2).clamp(-1+eps,1-eps).acos_()[...,None,None]%torch.pi # ln(R) will explode if theta==pi
        lnR = 1/(2*Lie.taylor_A(theta)+1e-8)*(R-R.transpose(-2,-1)) # FIXME: wei-chiu finds it weird
        w0,w1,w2 = lnR[...,2,1],lnR[...,0,2],lnR[...,1,0]
        w = torch.stack([w0,w1,w2],dim=-1)
        return w

    @staticmethod
    def se3_to_SE3(wu): # [...,3]
        w,u = wu.split([3,3],dim=-1)
        wx = Lie.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = Lie.taylor_A(theta)
        B = Lie.taylor_B(theta)
        C = Lie.taylor_C(theta)
        R = I+A*wx+B*wx@wx
        V = I+B*wx+C*wx@wx
        Rt = torch.cat([R,(V@u[...,None])],dim=-1)
        return Rt

    @staticmethod
    def SE3_to_se3(Rt,eps=1e-8): # [...,3,4]
        R,t = Rt.split([3,1],dim=-1)
        w = Lie.SO3_to_so3(R)
        wx = Lie.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = Lie.taylor_A(theta)
        B = Lie.taylor_B(theta)
        invV = I-0.5*wx+(1-A/(2*B))/(theta**2+eps)*wx@wx
        u = (invV@t)[...,0]
        wu = torch.cat([w,u],dim=-1)
        return wu    

    @staticmethod
    def skew_symmetric(w):
        w0,w1,w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                          torch.stack([w2,O,-w0],dim=-1),
                          torch.stack([-w1,w0,O],dim=-1)],dim=-2)
        return wx

    @staticmethod
    def taylor_A(x,nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i>0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    
    @staticmethod
    def taylor_B(x,nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    
    @staticmethod
    def taylor_C(x,nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

class Quaternion:
    """
    MIT License

    Copyright (c) 2021 Chen-Hsuan Lin

    Reference https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/main/LICENSE
    """
    @staticmethod
    def to_vector(r):
        w, x, y, z = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        norm = 2 * w / (w * w + y * y + z * z)
        v = torch.column_stack((norm * w - 1, norm * z, -norm * y))
        return v

    # from gaussian splatting    
    @staticmethod
    def to_mat(r):
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

        q = r / norm[:, None]

        R = torch.zeros((q.size(0), 3, 3), device='cuda')

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        return R