add_executable(basic_pricing basic_pricing.cpp)
target_link_libraries(basic_pricing PRIVATE OptionsPricingLib)

add_executable(portfolio_pricing portfolio_pricing.cpp)
target_link_libraries(portfolio_pricing PRIVATE OptionsPricingLib)

add_executable(real_time_demo real_time_demo.cpp)
target_link_libraries(real_time_demo PRIVATE OptionsPricingLib)