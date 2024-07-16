using DRLicenta.Data;
using DRLicenta.DTO;
using Microsoft.EntityFrameworkCore;
using System.Security.Cryptography;
using System.Text;

namespace DRLicenta.Services
{
    public class UserService : IUserService
    {
        private readonly UserDataContext _userDataContext;
        private readonly ILogger<UserService> _logger;

        public UserService(UserDataContext userDataContext, ILogger<UserService> logger)
        {
            _userDataContext = userDataContext;
            _logger = logger;
        }

        public async Task<string> Register(UserRegisterDto userRegisterDto)
        {
            if (await _userDataContext.Users.AnyAsync(u => u.Email == userRegisterDto.Email))
            {
                return "User already exists.";
            }

            using var hmac = new HMACSHA512();
            var user = new User
            {
                CNP = userRegisterDto.CNP,
                Email = userRegisterDto.Email,
                PhoneNumber = userRegisterDto.PhoneNumber,
                PasswordHash = Convert.ToBase64String(hmac.ComputeHash(Encoding.UTF8.GetBytes(userRegisterDto.Password))),
                PasswordSalt = Convert.ToBase64String(hmac.Key) // Store the salt
            };

            _userDataContext.Users.Add(user);
            await _userDataContext.SaveChangesAsync();
            return "User registered successfully.";
        }

        public async Task<string> Login(UserLoginDto userLoginDto)
        {
            _logger.LogInformation("Login attempt for email: {Email}", userLoginDto.Email);
            var user = await _userDataContext.Users.SingleOrDefaultAsync(u => u.Email == userLoginDto.Email);
            if (user == null)
            {
                _logger.LogWarning("User not found: {Email}", userLoginDto.Email);
                return "Invalid email or password.";
            }

            using var hmac = new HMACSHA512(Convert.FromBase64String(user.PasswordSalt));
            var computedHash = Convert.ToBase64String(hmac.ComputeHash(Encoding.UTF8.GetBytes(userLoginDto.Password)));

            if (computedHash != user.PasswordHash)
            {
                _logger.LogWarning("Invalid password for email: {Email}", userLoginDto.Email);
                return "Invalid email or password.";
            }

            _logger.LogInformation("Login successful for email: {Email}", userLoginDto.Email);
            return "Login successful.";
        }
    }
    
}
