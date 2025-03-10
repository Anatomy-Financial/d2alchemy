"""
Example SQLAlchemy models for testing the sqlalchemy-d2 library
"""

from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column, relationship

Base = declarative_base()


class User(Base):
    """User model representing application users"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime, nullable=False)

    # Relationships
    orders: Mapped[List["Order"]] = relationship("Order", back_populates="user")

    def __repr__(self) -> str:
        return f"<User(username='{self.username}', email='{self.email}')>"


class Product(Base):
    """Product model representing items for sale"""

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(500))
    price: Mapped[float] = mapped_column(Float, nullable=False)
    stock: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    order_items: Mapped[List["OrderItem"]] = relationship(
        "OrderItem", back_populates="product"
    )

    def __repr__(self) -> str:
        return f"<Product(name='{self.name}', price={self.price})>"


class Order(Base):
    """Order model representing customer purchases"""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    order_date: Mapped[DateTime] = mapped_column(DateTime, nullable=False)
    total: Mapped[float] = mapped_column(Float, nullable=False)

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="orders")
    items: Mapped[List["OrderItem"]] = relationship("OrderItem", back_populates="order")

    def __repr__(self) -> str:
        return f"<Order(id={self.id}, user_id={self.user_id}, total={self.total})>"


class OrderItem(Base):
    """OrderItem model representing items in an order"""

    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("orders.id"), nullable=False
    )
    product_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("products.id"), nullable=False
    )
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)

    # Relationships
    order: Mapped[Order] = relationship("Order", back_populates="items")
    product: Mapped[Product] = relationship("Product", back_populates="order_items")

    def __repr__(self) -> str:
        return f"<OrderItem(order_id={self.order_id}, product_id={self.product_id})>"
