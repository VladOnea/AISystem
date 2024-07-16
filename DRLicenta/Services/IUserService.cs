using DRLicenta.DTO;

namespace DRLicenta.Services
{
    public interface IUserService
    {
        Task<string> Register(UserRegisterDto userRegisterDto);
        Task<string> Login(UserLoginDto userLoginDto);
    }
}
