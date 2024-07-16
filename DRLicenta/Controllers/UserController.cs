using DRLicenta.Data;
using DRLicenta.DTO;
using DRLicenta.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace DRLicenta.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class UserController : ControllerBase
    {
        private readonly IUserService _userService;

        public UserController(IUserService userService)
        {
            _userService = userService;
        }

        [HttpPost("register")]
        public async Task<IActionResult> Register(UserRegisterDto userRegisterDto)
        {
            var result = await _userService.Register(userRegisterDto);
            if (result == "User already exists.")
            {
                return BadRequest(new { message = result });
            }

            return Ok(new { message = result });
        }

        [HttpPost("login")]
        public async Task<IActionResult> Login(UserLoginDto userLoginDto)
        {
            var result = await _userService.Login(userLoginDto);
            if (result == "Invalid email or password.")
            {
                return Unauthorized(new { message = result });
            }

            return Ok(new { message = result });
        }

    }
}
